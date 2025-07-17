#!/usr/bin/env python3
"""
AGENT 3 Performance Optimization System
Production-ready performance optimization for sub-millisecond inference
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import time
import numpy as np
from pathlib import Path
import gc
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict
import json
import psutil
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    meets_target: bool
    model_size_mb: float

class MemoryManager:
    """Advanced memory management for tensor operations"""
    
    def __init__(self, max_pool_size: int = 10000):
        self.tensor_pools = defaultdict(list)
        self.max_pool_size = max_pool_size
        self.lock = threading.Lock()
        self.stats = {
            'pool_hits': 0,
            'pool_misses': 0,
            'current_pool_size': 0
        }
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """Get tensor from pool or create new one"""
        key = (shape, dtype, device)
        
        with self.lock:
            if key in self.tensor_pools and self.tensor_pools[key]:
                tensor = self.tensor_pools[key].pop()
                tensor.zero_()
                self.stats['pool_hits'] += 1
                return tensor
        
        # Create new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.stats['pool_misses'] += 1
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if tensor.device.type == 'cpu':  # Only pool CPU tensors
            key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
            
            with self.lock:
                if len(self.tensor_pools[key]) < self.max_pool_size:
                    self.tensor_pools[key].append(tensor)
                    self.stats['current_pool_size'] += 1
    
    def cleanup(self):
        """Cleanup memory pools"""
        with self.lock:
            self.tensor_pools.clear()
            self.stats['current_pool_size'] = 0
        gc.collect()

class JITOptimizedTacticalActor(nn.Module):
    """JIT-optimized tactical actor without backward hooks"""
    
    def __init__(self, agent_id: str, input_size: int = 420, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        
        self.agent_id = agent_id
        self.input_size = input_size
        
        # Agent-specific feature weights (fixed buffers)
        if agent_id == "fvg":
            weights = torch.tensor([2.0, 2.0, 1.0, 0.5, 0.5, 0.3, 0.3]).repeat(60)
        elif agent_id == "momentum":
            weights = torch.tensor([0.3, 0.3, 0.5, 0.2, 0.2, 2.0, 2.0]).repeat(60)
        elif agent_id == "entry":
            weights = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]).repeat(60)
        else:
            weights = torch.ones(420)
        
        self.register_buffer('feature_weights', weights)
        
        # Optimized network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities only (JIT compatible)"""
        # Flatten input
        if state.dim() == 3:
            x = state.view(state.size(0), -1)
        else:
            x = state.view(1, -1)
        
        # Apply feature weights
        x = x * self.feature_weights
        
        # Network forward
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

class JITOptimizedStrategicActor(nn.Module):
    """JIT-optimized strategic actor"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities only"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

class JITOptimizedCritic(nn.Module):
    """JIT-optimized critic"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value only"""
        return self.network(state).squeeze(-1)

class JITOptimizedTacticalSystem(nn.Module):
    """JIT-optimized tactical system"""
    
    def __init__(self, input_size: int = 420, hidden_dim: int = 128):
        super().__init__()
        
        self.fvg_agent = JITOptimizedTacticalActor("fvg", input_size, hidden_dim)
        self.momentum_agent = JITOptimizedTacticalActor("momentum", input_size, hidden_dim)
        self.entry_agent = JITOptimizedTacticalActor("entry", input_size, hidden_dim)
        self.critic = JITOptimizedCritic(input_size, 64)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tuple of (fvg_probs, momentum_probs, entry_probs, value)"""
        fvg_probs = self.fvg_agent(state)
        momentum_probs = self.momentum_agent(state)
        entry_probs = self.entry_agent(state)
        
        # Flatten state for critic
        if state.dim() == 3:
            state_flat = state.view(state.size(0), -1)
        else:
            state_flat = state.view(1, -1)
        
        value = self.critic(state_flat)
        
        return fvg_probs, momentum_probs, entry_probs, value

class JITOptimizedStrategicSystem(nn.Module):
    """JIT-optimized strategic system"""
    
    def __init__(self):
        super().__init__()
        
        self.mlmi_agent = JITOptimizedStrategicActor(input_dim=4, hidden_dim=64)
        self.nwrqk_agent = JITOptimizedStrategicActor(input_dim=6, hidden_dim=64)
        self.mmd_agent = JITOptimizedStrategicActor(input_dim=3, hidden_dim=32)
        self.critic = JITOptimizedCritic(state_dim=13, hidden_dim=64)
    
    def forward(self, mlmi_state: torch.Tensor, nwrqk_state: torch.Tensor, mmd_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tuple of (mlmi_probs, nwrqk_probs, mmd_probs, value)"""
        mlmi_probs = self.mlmi_agent(mlmi_state)
        nwrqk_probs = self.nwrqk_agent(nwrqk_state)
        mmd_probs = self.mmd_agent(mmd_state)
        
        # Combine states for critic
        combined_state = torch.cat([mlmi_state, nwrqk_state, mmd_state], dim=-1)
        value = self.critic(combined_state)
        
        return mlmi_probs, nwrqk_probs, mmd_probs, value

class ModelQuantizer:
    """Model quantization for inference optimization"""
    
    def __init__(self):
        self.quantization_stats = {}
    
    def quantize_model(self, model: nn.Module, model_name: str) -> nn.Module:
        """Apply dynamic quantization"""
        logger.info(f"Quantizing {model_name}")
        
        try:
            model.eval()
            
            # Dynamic quantization for CPU
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            # Calculate size reduction
            original_size = sum(p.numel() * 4 for p in model.parameters())
            quantized_size = original_size // 4  # Rough estimate
            
            self.quantization_stats[model_name] = {
                'original_size_mb': original_size / (1024 * 1024),
                'quantized_size_mb': quantized_size / (1024 * 1024),
                'compression_ratio': original_size / quantized_size
            }
            
            logger.info(f"Quantized {model_name}: {original_size//1024}KB -> {quantized_size//1024}KB")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed for {model_name}: {e}")
            return model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quantization statistics"""
        return self.quantization_stats

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.memory_manager = MemoryManager()
        self.quantizer = ModelQuantizer()
        
        self.compiled_models = {}
        self.quantized_models = {}
        self.performance_results = {}
        
        # Performance monitoring
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.performance_history = []
        
        logger.info(f"PerformanceOptimizer initialized on {device}")
    
    def create_optimized_models(self) -> Dict[str, nn.Module]:
        """Create all optimized models"""
        logger.info("Creating optimized models")
        
        models = {
            'tactical_system': JITOptimizedTacticalSystem(),
            'strategic_system': JITOptimizedStrategicSystem(),
            'fvg_agent': JITOptimizedTacticalActor("fvg"),
            'momentum_agent': JITOptimizedTacticalActor("momentum"),
            'entry_agent': JITOptimizedTacticalActor("entry"),
            'mlmi_agent': JITOptimizedStrategicActor(input_dim=4),
            'nwrqk_agent': JITOptimizedStrategicActor(input_dim=6),
            'mmd_agent': JITOptimizedStrategicActor(input_dim=3),
            'critic': JITOptimizedCritic(state_dim=13)
        }
        
        # Set all to eval mode
        for model in models.values():
            model.eval()
        
        return models
    
    def compile_models_jit(self, models: Dict[str, nn.Module]) -> Dict[str, torch.jit.ScriptModule]:
        """Compile models using TorchScript"""
        logger.info("Compiling models with JIT")
        
        compiled_models = {}
        
        # Example inputs
        tactical_input = torch.randn(1, 60, 7)
        mlmi_input = torch.randn(1, 4)
        nwrqk_input = torch.randn(1, 6)
        mmd_input = torch.randn(1, 3)
        critic_input = torch.randn(1, 13)
        
        # Compile tactical system
        try:
            with torch.no_grad():
                traced_tactical = torch.jit.trace(models['tactical_system'], tactical_input)
                traced_tactical = torch.jit.optimize_for_inference(traced_tactical)
                compiled_models['tactical_system'] = traced_tactical
                logger.info("✅ Compiled tactical system")
        except Exception as e:
            logger.error(f"❌ Failed to compile tactical system: {e}")
            compiled_models['tactical_system'] = models['tactical_system']
        
        # Compile strategic system
        try:
            with torch.no_grad():
                traced_strategic = torch.jit.trace(models['strategic_system'], (mlmi_input, nwrqk_input, mmd_input))
                traced_strategic = torch.jit.optimize_for_inference(traced_strategic)
                compiled_models['strategic_system'] = traced_strategic
                logger.info("✅ Compiled strategic system")
        except Exception as e:
            logger.error(f"❌ Failed to compile strategic system: {e}")
            compiled_models['strategic_system'] = models['strategic_system']
        
        # Compile individual agents
        agent_inputs = {
            'fvg_agent': tactical_input,
            'momentum_agent': tactical_input,
            'entry_agent': tactical_input,
            'mlmi_agent': mlmi_input,
            'nwrqk_agent': nwrqk_input,
            'mmd_agent': mmd_input,
            'critic': critic_input
        }
        
        for agent_name, agent_model in models.items():
            if agent_name in ['tactical_system', 'strategic_system']:
                continue
            
            try:
                with torch.no_grad():
                    traced_agent = torch.jit.trace(agent_model, agent_inputs[agent_name])
                    traced_agent = torch.jit.optimize_for_inference(traced_agent)
                    compiled_models[agent_name] = traced_agent
                    logger.info(f"✅ Compiled {agent_name}")
            except Exception as e:
                logger.error(f"❌ Failed to compile {agent_name}: {e}")
                compiled_models[agent_name] = agent_model
        
        self.compiled_models = compiled_models
        return compiled_models
    
    def quantize_models(self, models: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
        """Quantize models for production"""
        logger.info("Quantizing models")
        
        quantized_models = {}
        for model_name, model in models.items():
            quantized_models[model_name] = self.quantizer.quantize_model(model, model_name)
        
        self.quantized_models = quantized_models
        return quantized_models
    
    def benchmark_models(self, models: Dict[str, Any], num_iterations: int = 1000) -> Dict[str, PerformanceMetrics]:
        """Benchmark model performance"""
        logger.info(f"Benchmarking models with {num_iterations} iterations")
        
        results = {}
        
        # Test inputs
        tactical_input = torch.randn(1, 60, 7)
        mlmi_input = torch.randn(1, 4)
        nwrqk_input = torch.randn(1, 6)
        mmd_input = torch.randn(1, 3)
        critic_input = torch.randn(1, 13)
        
        for model_name, model in models.items():
            # Determine input type
            if 'tactical' in model_name or model_name in ['fvg_agent', 'momentum_agent', 'entry_agent']:
                test_input = tactical_input
            elif model_name == 'strategic_system':
                test_input = (mlmi_input, nwrqk_input, mmd_input)
            elif model_name == 'mlmi_agent':
                test_input = mlmi_input
            elif model_name == 'nwrqk_agent':
                test_input = nwrqk_input
            elif model_name == 'mmd_agent':
                test_input = mmd_input
            else:  # critic
                test_input = critic_input
            
            # Warm up
            with torch.no_grad():
                for _ in range(20):
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
            
            # Benchmark
            latencies = []
            memory_usage = []
            
            with torch.no_grad():
                for i in range(num_iterations):
                    # Memory cleanup every 100 iterations
                    if i % 100 == 0:
                        gc.collect()
                    
                    # Track memory
                    mem_before = psutil.virtual_memory().used
                    
                    start_time = time.perf_counter()
                    
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
                    
                    end_time = time.perf_counter()
                    
                    mem_after = psutil.virtual_memory().used
                    
                    latencies.append((end_time - start_time) * 1000)
                    memory_usage.append((mem_after - mem_before) / (1024 * 1024))
            
            # Calculate model size
            model_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            
            results[model_name] = PerformanceMetrics(
                avg_latency_ms=avg_latency,
                p99_latency_ms=p99_latency,
                throughput_qps=1000 / avg_latency,
                memory_usage_mb=np.mean(memory_usage),
                cpu_usage_percent=psutil.cpu_percent(),
                meets_target=p99_latency < 1.0,  # <1ms target
                model_size_mb=model_size
            )
        
        self.performance_results = results
        return results
    
    def save_models(self, models: Dict[str, Any], save_dir: Path = Path("models/optimized")):
        """Save optimized models"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            model_path = save_dir / f"{model_name}.pt"
            
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:  # JIT model
                torch.jit.save(model, model_path)
            
            logger.info(f"💾 Saved {model_name} to {model_path}")
        
        # Save performance results
        results_path = save_dir / "performance_results.json"
        with open(results_path, 'w') as f:
            results_dict = {}
            for model_name, metrics in self.performance_results.items():
                results_dict[model_name] = {
                    'avg_latency_ms': float(metrics.avg_latency_ms),
                    'p99_latency_ms': float(metrics.p99_latency_ms),
                    'throughput_qps': float(metrics.throughput_qps),
                    'memory_usage_mb': float(metrics.memory_usage_mb),
                    'cpu_usage_percent': float(metrics.cpu_usage_percent),
                    'meets_target': bool(metrics.meets_target),
                    'model_size_mb': float(metrics.model_size_mb)
                }
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"📊 Saved performance results to {results_path}")

class DistributedInferenceServer:
    """Distributed inference server with load balancing"""
    
    def __init__(self, model_dir: Path, num_workers: int = 4):
        self.model_dir = Path(model_dir)
        self.num_workers = num_workers
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Load balancing
        self.request_count = 0
        self.worker_stats = defaultdict(lambda: {'requests': 0, 'total_time': 0.0})
        
        # Caching
        self.cache = {}
        self.cache_max_size = 10000
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def load_models(self):
        """Load optimized models"""
        logger.info("Loading models for distributed inference")
        
        model_files = list(self.model_dir.glob("*.pt"))
        for model_file in model_files:
            model_name = model_file.stem
            try:
                if model_file.suffix == '.pt':
                    model = torch.jit.load(str(model_file))
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"✅ Loaded {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
    
    def infer(self, model_name: str, input_data: torch.Tensor, use_cache: bool = True) -> Dict[str, Any]:
        """Perform inference with caching"""
        # Check cache
        if use_cache:
            cache_key = f"{model_name}_{hash(input_data.numpy().tobytes())}"
            if cache_key in self.cache:
                self.cache_stats['hits'] += 1
                return self.cache[cache_key]
        
        # Perform inference
        start_time = time.perf_counter()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        with torch.no_grad():
            result = self.models[model_name](input_data)
        
        end_time = time.perf_counter()
        
        inference_result = {
            'result': result,
            'inference_time_ms': (end_time - start_time) * 1000,
            'model_name': model_name,
            'cached': False
        }
        
        # Cache result
        if use_cache:
            if len(self.cache) >= self.cache_max_size:
                # Simple LRU
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = inference_result
            self.cache_stats['misses'] += 1
        
        return inference_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'models_loaded': list(self.models.keys()),
            'cache_stats': self.cache_stats,
            'worker_stats': dict(self.worker_stats),
            'cache_hit_rate': self.cache_stats['hits'] / max(sum(self.cache_stats.values()), 1)
        }

class ContinuousPerformanceMonitor:
    """Continuous performance monitoring and regression detection"""
    
    def __init__(self, alert_threshold_ms: float = 1.0):
        self.alert_threshold_ms = alert_threshold_ms
        self.performance_history = []
        self.alerts = []
        
        # Performance baselines
        self.baselines = {}
        self.monitoring_active = False
    
    def record_performance(self, model_name: str, latency_ms: float, throughput_qps: float, memory_mb: float):
        """Record performance metrics"""
        timestamp = time.time()
        
        record = {
            'timestamp': timestamp,
            'model_name': model_name,
            'latency_ms': latency_ms,
            'throughput_qps': throughput_qps,
            'memory_mb': memory_mb
        }
        
        self.performance_history.append(record)
        
        # Check for performance regression
        if self.monitoring_active:
            self.check_regression(record)
    
    def check_regression(self, record: Dict[str, Any]):
        """Check for performance regression"""
        model_name = record['model_name']
        current_latency = record['latency_ms']
        
        if model_name in self.baselines:
            baseline_latency = self.baselines[model_name]['latency_ms']
            
            # Alert if current latency is 20% worse than baseline
            if current_latency > baseline_latency * 1.2:
                alert = {
                    'timestamp': record['timestamp'],
                    'model_name': model_name,
                    'type': 'performance_regression',
                    'current_latency_ms': current_latency,
                    'baseline_latency_ms': baseline_latency,
                    'degradation_percent': ((current_latency - baseline_latency) / baseline_latency) * 100
                }
                self.alerts.append(alert)
                logger.warning(f"Performance regression detected for {model_name}: {current_latency:.2f}ms vs {baseline_latency:.2f}ms baseline")
        
        # Alert if absolute threshold exceeded
        if current_latency > self.alert_threshold_ms:
            alert = {
                'timestamp': record['timestamp'],
                'model_name': model_name,
                'type': 'threshold_exceeded',
                'current_latency_ms': current_latency,
                'threshold_ms': self.alert_threshold_ms
            }
            self.alerts.append(alert)
            logger.warning(f"Latency threshold exceeded for {model_name}: {current_latency:.2f}ms > {self.alert_threshold_ms:.2f}ms")
    
    def set_baselines(self, baseline_metrics: Dict[str, PerformanceMetrics]):
        """Set performance baselines"""
        for model_name, metrics in baseline_metrics.items():
            self.baselines[model_name] = {
                'latency_ms': metrics.avg_latency_ms,
                'throughput_qps': metrics.throughput_qps,
                'memory_mb': metrics.memory_usage_mb
            }
        
        self.monitoring_active = True
        logger.info("Performance baselines set and monitoring activated")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        return self.alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {}
        
        summary = {}
        for record in self.performance_history[-100:]:  # Last 100 records
            model_name = record['model_name']
            if model_name not in summary:
                summary[model_name] = {
                    'latencies': [],
                    'throughputs': [],
                    'memory_usage': []
                }
            
            summary[model_name]['latencies'].append(record['latency_ms'])
            summary[model_name]['throughputs'].append(record['throughput_qps'])
            summary[model_name]['memory_usage'].append(record['memory_mb'])
        
        # Calculate statistics
        for model_name, data in summary.items():
            summary[model_name] = {
                'avg_latency_ms': np.mean(data['latencies']),
                'p99_latency_ms': np.percentile(data['latencies'], 99),
                'avg_throughput_qps': np.mean(data['throughputs']),
                'avg_memory_mb': np.mean(data['memory_usage'])
            }
        
        return summary

def main():
    """Main performance optimization workflow"""
    print("🚀 GRAND MODEL PERFORMANCE OPTIMIZATION SYSTEM")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(device="cpu")
    
    # Create optimized models
    print("\n📦 Creating optimized models...")
    models = optimizer.create_optimized_models()
    
    # Compile with JIT
    print("\n⚡ Compiling models with JIT...")
    compiled_models = optimizer.compile_models_jit(models)
    
    # Quantize models
    print("\n🔧 Quantizing models...")
    quantized_models = optimizer.quantize_models(models)
    
    # Benchmark performance
    print("\n📊 Benchmarking performance...")
    jit_results = optimizer.benchmark_models(compiled_models)
    
    optimizer.performance_results = jit_results  # Update results
    
    # Save models
    print("\n💾 Saving optimized models...")
    optimizer.save_models(compiled_models)
    
    # Performance summary
    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE OPTIMIZATION RESULTS")
    print("=" * 80)
    
    systems_meeting_target = 0
    
    for model_name, metrics in jit_results.items():
        status = "✅ PASS" if metrics.meets_target else "❌ FAIL"
        print(f"{model_name:25}: {metrics.p99_latency_ms:6.2f}ms p99 | {metrics.throughput_qps:8.1f} QPS | {metrics.model_size_mb:6.2f}MB {status}")
        
        if 'system' in model_name and metrics.meets_target:
            systems_meeting_target += 1
    
    print("\n" + "=" * 80)
    print("🏆 FINAL ASSESSMENT")
    print("=" * 80)
    
    tactical_metrics = jit_results.get('tactical_system')
    strategic_metrics = jit_results.get('strategic_system')
    
    if tactical_metrics and tactical_metrics.meets_target:
        print(f"✅ TACTICAL SYSTEM: {tactical_metrics.p99_latency_ms:.2f}ms - SUB-MILLISECOND READY")
    else:
        print(f"❌ TACTICAL SYSTEM: {tactical_metrics.p99_latency_ms if tactical_metrics else 'N/A'}ms - NEEDS OPTIMIZATION")
    
    if strategic_metrics and strategic_metrics.meets_target:
        print(f"✅ STRATEGIC SYSTEM: {strategic_metrics.p99_latency_ms:.2f}ms - SUB-MILLISECOND READY")
    else:
        print(f"❌ STRATEGIC SYSTEM: {strategic_metrics.p99_latency_ms if strategic_metrics else 'N/A'}ms - NEEDS OPTIMIZATION")
    
    if systems_meeting_target == 2:
        print("\n🎉 SUCCESS: ALL SYSTEMS ACHIEVE SUB-MILLISECOND INFERENCE!")
        print("🚀 Models are ready for high-frequency trading deployment")
    else:
        print(f"\n⚠️ WARNING: {2-systems_meeting_target} system(s) still need optimization")
    
    print("=" * 80)
    
    # Initialize monitoring
    monitor = ContinuousPerformanceMonitor(alert_threshold_ms=1.0)
    monitor.set_baselines(jit_results)
    
    print("\n🔍 Continuous performance monitoring activated")
    print("💡 Models saved to models/optimized/")
    print("📊 Performance results saved to models/optimized/performance_results.json")

if __name__ == "__main__":
    main()