#!/usr/bin/env python3
"""
JIT Optimized Engine for Critical Path Performance
Implements TorchScript compilation, vectorization, and optimized execution paths
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
import gc
import psutil
import math
from contextlib import contextmanager
from functools import wraps, lru_cache
import weakref
from collections import defaultdict
import concurrent.futures
import asyncio

from .advanced_caching_system import MultiLevelCache, CacheKeyGenerator, get_global_cache

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    enable_jit: bool = True
    enable_quantization: bool = True
    enable_fusion: bool = True
    enable_vectorization: bool = True
    enable_async: bool = True
    max_batch_size: int = 32
    prefetch_size: int = 4
    memory_efficient: bool = True
    optimization_level: str = 'O2'  # O0, O1, O2, O3
    target_latency_ms: float = 1.0

class JITModelWrapper:
    """Wrapper for JIT-compiled models with optimization features"""
    
    def __init__(self, model: nn.Module, optimization_config: OptimizationConfig):
        self.original_model = model
        self.config = optimization_config
        self.compiled_model = None
        self.quantized_model = None
        self.cache = get_global_cache()
        
        # Performance metrics
        self.inference_times = []
        self.compilation_time = 0.0
        self.optimization_stats = {
            'total_inferences': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0,
            'model_size_mb': 0.0
        }
        
        # Warm-up cache
        self.warmup_cache = {}
        
        # Compile model
        self._compile_model()
    
    def _compile_model(self):
        """Compile model with JIT and optimizations"""
        start_time = time.time()
        
        try:
            # Set model to eval mode
            self.original_model.eval()
            
            if self.config.enable_jit:
                # Create example inputs for tracing
                example_inputs = self._create_example_inputs()
                
                # Trace the model
                with torch.no_grad():
                    if isinstance(example_inputs, (list, tuple)):
                        traced_model = torch.jit.trace(self.original_model, example_inputs)
                    else:
                        traced_model = torch.jit.trace(self.original_model, example_inputs)
                
                # Optimize for inference
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                # Enable fusion if requested
                if self.config.enable_fusion:
                    traced_model = self._enable_fusion(traced_model)
                
                self.compiled_model = traced_model
                logger.info(f"Successfully compiled model with JIT")
            
            # Quantization
            if self.config.enable_quantization:
                self.quantized_model = self._quantize_model()
                logger.info(f"Successfully quantized model")
            
            self.compilation_time = time.time() - start_time
            
            # Calculate model size
            self.optimization_stats['model_size_mb'] = self._calculate_model_size()
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            self.compiled_model = self.original_model
    
    def _create_example_inputs(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Create example inputs for tracing"""
        # This needs to be customized based on the model
        # For now, create generic inputs
        if hasattr(self.original_model, 'input_dim'):
            return torch.randn(1, self.original_model.input_dim)
        else:
            # Default tactical input
            return torch.randn(1, 60, 7)
    
    def _enable_fusion(self, traced_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Enable operator fusion for better performance"""
        try:
            # Enable conv-relu fusion
            traced_model = torch.jit.freeze(traced_model)
            
            # Additional fusion optimizations
            if hasattr(torch.jit, 'fuse_modules'):
                traced_model = torch.jit.fuse_modules(traced_model)
            
            return traced_model
        except Exception as e:
            logger.warning(f"Fusion optimization failed: {e}")
            return traced_model
    
    def _quantize_model(self) -> nn.Module:
        """Apply dynamic quantization"""
        try:
            model_to_quantize = self.compiled_model or self.original_model
            
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model_to_quantize,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return self.compiled_model or self.original_model
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        model = self.quantized_model or self.compiled_model or self.original_model
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        return size_mb
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Optimized forward pass"""
        # Generate cache key
        cache_key = CacheKeyGenerator.generate_key(
            {'args': args, 'kwargs': kwargs}, 
            prefix=f"jit_forward_{id(self)}"
        )
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.optimization_stats['cache_hits'] += 1
            return cached_result
        
        # Perform inference
        start_time = time.time()
        
        with torch.no_grad():
            # Use the best available model
            model = self.quantized_model or self.compiled_model or self.original_model
            
            if isinstance(args, (list, tuple)) and len(args) > 0:
                result = model(*args, **kwargs)
            else:
                result = model(args, **kwargs)
        
        inference_time = time.time() - start_time
        
        # Update statistics
        self.inference_times.append(inference_time)
        self.optimization_stats['total_inferences'] += 1
        
        # Keep only last 1000 measurements
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        
        self.optimization_stats['avg_inference_time'] = np.mean(self.inference_times)
        
        # Cache result if inference time is significant
        if inference_time > 0.001:  # 1ms threshold
            self.cache.put(cache_key, result, ttl=300)  # 5 minute TTL
        
        return result
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable"""
        return self.forward(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.optimization_stats,
            'compilation_time': self.compilation_time,
            'is_jit_compiled': self.compiled_model is not None,
            'is_quantized': self.quantized_model is not None,
            'inference_times_p50': np.percentile(self.inference_times, 50) if self.inference_times else 0,
            'inference_times_p95': np.percentile(self.inference_times, 95) if self.inference_times else 0,
            'inference_times_p99': np.percentile(self.inference_times, 99) if self.inference_times else 0,
        }
    
    def warmup(self, num_iterations: int = 100):
        """Warm up the model for consistent performance"""
        logger.info(f"Warming up model for {num_iterations} iterations...")
        
        example_inputs = self._create_example_inputs()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if isinstance(example_inputs, (list, tuple)):
                    self.forward(*example_inputs)
                else:
                    self.forward(example_inputs)
        
        logger.info(f"Model warmup complete")

class VectorizedOperations:
    """Optimized vectorized operations for common computations"""
    
    @staticmethod
    @torch.jit.script
    def batch_matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorized batch matrix multiplication"""
        return torch.bmm(a, b)
    
    @staticmethod
    @torch.jit.script
    def batch_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Vectorized batch attention computation"""
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        d_k = query.size(-1)
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attention_weights, value)
        
        return output
    
    @staticmethod
    @torch.jit.script
    def batch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                        eps: float = 1e-5) -> torch.Tensor:
        """Vectorized batch layer normalization"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return weight * (x - mean) / (std + eps) + bias
    
    @staticmethod
    @torch.jit.script
    def batch_gelu(x: torch.Tensor) -> torch.Tensor:
        """Vectorized batch GELU activation"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    @staticmethod
    @torch.jit.script
    def batch_feature_scaling(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Vectorized batch feature scaling"""
        return x * scale + shift
    
    @staticmethod
    @torch.jit.script
    def batch_temporal_pooling(x: torch.Tensor, pool_type: str = "mean") -> torch.Tensor:
        """Vectorized batch temporal pooling"""
        if pool_type == "mean":
            return x.mean(dim=1)
        elif pool_type == "max":
            return x.max(dim=1)[0]
        elif pool_type == "sum":
            return x.sum(dim=1)
        else:
            return x.mean(dim=1)

class AsyncInferenceEngine:
    """Asynchronous inference engine for non-blocking operations"""
    
    def __init__(self, model: JITModelWrapper, max_workers: int = 4):
        self.model = model
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_requests = {}
        self.batch_queue = asyncio.Queue(maxsize=100)
        self.batch_processor_task = None
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0
        }
        self.lock = asyncio.Lock()
    
    async def infer_async(self, input_data: torch.Tensor, timeout: float = 5.0) -> torch.Tensor:
        """Perform asynchronous inference"""
        loop = asyncio.get_event_loop()
        
        # Create future for result
        future = loop.create_future()
        
        # Submit to batch queue
        await self.batch_queue.put((input_data, future))
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Inference timeout after {timeout}s")
            raise
    
    async def start_batch_processor(self):
        """Start the batch processor"""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
    
    async def stop_batch_processor(self):
        """Stop the batch processor"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
    
    async def _batch_processor(self):
        """Process inference requests in batches"""
        batch_size = 8
        batch_timeout = 0.01  # 10ms timeout
        
        while True:
            batch = []
            futures = []
            
            try:
                # Collect batch
                deadline = asyncio.get_event_loop().time() + batch_timeout
                
                while len(batch) < batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        input_data, future = await asyncio.wait_for(
                            self.batch_queue.get(), 
                            timeout=remaining_time
                        )
                        batch.append(input_data)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                start_time = time.time()
                
                # Stack inputs for batch processing
                batch_tensor = torch.stack(batch)
                
                # Perform inference
                loop = asyncio.get_event_loop()
                batch_result = await loop.run_in_executor(
                    self.executor, 
                    self.model.forward, 
                    batch_tensor
                )
                
                processing_time = time.time() - start_time
                
                # Update stats
                async with self.lock:
                    self.stats['total_requests'] += len(batch)
                    self.stats['batched_requests'] += 1
                    self.stats['avg_batch_size'] = (
                        self.stats['avg_batch_size'] * (self.stats['batched_requests'] - 1) + len(batch)
                    ) / self.stats['batched_requests']
                    self.stats['avg_processing_time'] = (
                        self.stats['avg_processing_time'] * (self.stats['batched_requests'] - 1) + processing_time
                    ) / self.stats['batched_requests']
                
                # Return results to futures
                for i, future in enumerate(futures):
                    if not future.cancelled():
                        future.set_result(batch_result[i])
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set exception for all futures
                for future in futures:
                    if not future.cancelled():
                        future.set_exception(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get async inference statistics"""
        async with self.lock:
            return self.stats.copy()

class OptimizedModelFactory:
    """Factory for creating optimized models"""
    
    @staticmethod
    def create_optimized_tactical_system(input_size: int = 420, 
                                       hidden_dim: int = 128,
                                       config: OptimizationConfig = None) -> JITModelWrapper:
        """Create optimized tactical system"""
        if config is None:
            config = OptimizationConfig()
        
        # Create base model
        model = OptimizedTacticalSystem(input_size, hidden_dim)
        
        # Wrap with JIT optimization
        return JITModelWrapper(model, config)
    
    @staticmethod
    def create_optimized_strategic_system(config: OptimizationConfig = None) -> JITModelWrapper:
        """Create optimized strategic system"""
        if config is None:
            config = OptimizationConfig()
        
        # Create base model
        model = OptimizedStrategicSystem()
        
        # Wrap with JIT optimization
        return JITModelWrapper(model, config)
    
    @staticmethod
    def create_optimized_shared_policy(input_dim: int = 136,
                                     config: OptimizationConfig = None) -> JITModelWrapper:
        """Create optimized shared policy"""
        if config is None:
            config = OptimizationConfig()
        
        # Create base model
        model = OptimizedSharedPolicy(input_dim)
        
        # Wrap with JIT optimization
        return JITModelWrapper(model, config)

class OptimizedTacticalSystem(nn.Module):
    """JIT-optimized tactical system"""
    
    def __init__(self, input_size: int = 420, hidden_dim: int = 128):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Agent networks
        self.fvg_agent = self._create_agent_network("fvg")
        self.momentum_agent = self._create_agent_network("momentum")
        self.entry_agent = self._create_agent_network("entry")
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature importance weights (learnable)
        self.register_buffer('fvg_weights', torch.tensor([2.0, 2.0, 1.0, 0.5, 0.5, 0.3, 0.3]).repeat(60))
        self.register_buffer('momentum_weights', torch.tensor([0.3, 0.3, 0.5, 0.2, 0.2, 2.0, 2.0]).repeat(60))
        self.register_buffer('entry_weights', torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]).repeat(60))
        
        # Initialize weights
        self._init_weights()
    
    def _create_agent_network(self, agent_type: str) -> nn.Module:
        """Create agent-specific network"""
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3)  # 3 actions
        )
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Flatten input if needed
        if state.dim() == 3:
            batch_size = state.size(0)
            state = state.view(batch_size, -1)
        else:
            state = state.view(1, -1)
        
        # Apply feature weights
        fvg_state = state * self.fvg_weights
        momentum_state = state * self.momentum_weights
        entry_state = state * self.entry_weights
        
        # Agent forward passes
        fvg_logits = self.fvg_agent(fvg_state)
        momentum_logits = self.momentum_agent(momentum_state)
        entry_logits = self.entry_agent(entry_state)
        
        # Convert to probabilities
        fvg_probs = F.softmax(fvg_logits, dim=-1)
        momentum_probs = F.softmax(momentum_logits, dim=-1)
        entry_probs = F.softmax(entry_logits, dim=-1)
        
        # Critic value
        value = self.critic(state).squeeze(-1)
        
        return fvg_probs, momentum_probs, entry_probs, value

class OptimizedStrategicSystem(nn.Module):
    """JIT-optimized strategic system"""
    
    def __init__(self):
        super().__init__()
        
        # Agent networks
        self.mlmi_agent = self._create_agent_network(4, 64)
        self.nwrqk_agent = self._create_agent_network(6, 64)
        self.mmd_agent = self._create_agent_network(3, 32)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(13, 64),  # 4 + 6 + 3
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_agent_network(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create agent network"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # 3 actions
        )
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, mlmi_state: torch.Tensor, nwrqk_state: torch.Tensor, 
                mmd_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Agent forward passes
        mlmi_logits = self.mlmi_agent(mlmi_state)
        nwrqk_logits = self.nwrqk_agent(nwrqk_state)
        mmd_logits = self.mmd_agent(mmd_state)
        
        # Convert to probabilities
        mlmi_probs = F.softmax(mlmi_logits, dim=-1)
        nwrqk_probs = F.softmax(nwrqk_logits, dim=-1)
        mmd_probs = F.softmax(mmd_logits, dim=-1)
        
        # Combine states for critic
        combined_state = torch.cat([mlmi_state, nwrqk_state, mmd_state], dim=-1)
        value = self.critic(combined_state).squeeze(-1)
        
        return mlmi_probs, nwrqk_probs, mmd_probs, value

class OptimizedSharedPolicy(nn.Module):
    """JIT-optimized shared policy network"""
    
    def __init__(self, input_dim: int = 136):
        super().__init__()
        self.input_dim = input_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # 2 actions
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Policy output
        policy_logits = self.policy_net(state)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Value output
        value = self.value_net(state).squeeze(-1)
        
        return policy_probs, value

class PerformanceProfiler:
    """Performance profiler for optimization analysis"""
    
    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.active_profiles = {}
        self.lock = threading.RLock()
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self.lock:
                self.profiling_data[operation_name].append({
                    'duration': duration,
                    'memory_delta': memory_delta,
                    'timestamp': start_time
                })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        with self.lock:
            stats = {}
            
            for operation, measurements in self.profiling_data.items():
                if measurements:
                    durations = [m['duration'] for m in measurements]
                    memory_deltas = [m['memory_delta'] for m in measurements]
                    
                    stats[operation] = {
                        'count': len(measurements),
                        'avg_duration': np.mean(durations),
                        'p95_duration': np.percentile(durations, 95),
                        'p99_duration': np.percentile(durations, 99),
                        'avg_memory_delta': np.mean(memory_deltas),
                        'total_time': sum(durations)
                    }
            
            return stats
    
    def reset(self):
        """Reset profiling data"""
        with self.lock:
            self.profiling_data.clear()

# Global profiler instance
global_profiler = PerformanceProfiler()

# Decorators for performance monitoring
def profile_function(operation_name: str = None):
    """Decorator for profiling functions"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with global_profiler.profile(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def jit_compile(func):
    """Decorator for JIT compilation"""
    try:
        return torch.jit.script(func)
    except Exception as e:
        logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
        return func

# Example usage and testing
if __name__ == "__main__":
    # Test optimization
    config = OptimizationConfig(
        enable_jit=True,
        enable_quantization=True,
        enable_vectorization=True,
        target_latency_ms=1.0
    )
    
    # Create optimized tactical system
    tactical_system = OptimizedModelFactory.create_optimized_tactical_system(config=config)
    
    # Warm up
    tactical_system.warmup(100)
    
    # Test inference
    test_input = torch.randn(1, 60, 7)
    
    # Profile inference
    with global_profiler.profile("tactical_inference"):
        result = tactical_system(test_input)
    
    # Print results
    print(f"Tactical system stats: {tactical_system.get_stats()}")
    print(f"Profiler stats: {global_profiler.get_stats()}")
    
    logger.info("JIT optimization system test completed")