"""
Performance Optimization Engine for Intelligence Components.

Implements JIT compilation, memory pool management, batch processing optimizations,
and other performance enhancements to achieve <5ms inference targets.
"""

import torch
import torch.nn as nn
import torch.jit
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
import threading
from functools import lru_cache
import gc
import weakref

class JITOptimizer:
    """
    JIT compilation optimizer for PyTorch models and functions.
    
    Provides automatic JIT compilation with fallback mechanisms and
    performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # JIT compilation settings
        self.enable_jit = config.get('enable_jit', True)
        self.jit_warmup_iterations = config.get('jit_warmup_iterations', 10)
        self.fallback_on_failure = config.get('fallback_on_failure', True)
        
        # Compiled model cache
        self.compiled_models = {}
        self.compilation_failures = set()
        
        # Performance tracking
        self.compilation_times = {}
        self.execution_times = {}
        
        self.logger.info("JIT Optimizer initialized")
    
    def compile_model(
        self, 
        model: nn.Module, 
        example_inputs: Union[torch.Tensor, tuple], 
        model_name: str
    ) -> Optional[torch.jit.ScriptModule]:
        """
        Compile a PyTorch model with JIT optimization.
        
        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing
            model_name: Name for tracking and caching
            
        Returns:
            Compiled model or None if compilation failed
        """
        if not self.enable_jit or model_name in self.compilation_failures:
            return None
        
        if model_name in self.compiled_models:
            return self.compiled_models[model_name]
        
        try:
            self.logger.info(f"Compiling model: {model_name}")
            start_time = time.perf_counter()
            
            # Set model to eval mode for compilation
            model.eval()
            
            # Trace the model
            with torch.no_grad():
                if isinstance(example_inputs, (list, tuple)):
                    compiled_model = torch.jit.trace(model, example_inputs)
                else:
                    compiled_model = torch.jit.trace(model, (example_inputs,))
            
            # Warm up the compiled model
            self._warmup_compiled_model(compiled_model, example_inputs)
            
            # Cache the compiled model
            self.compiled_models[model_name] = compiled_model
            
            compilation_time = time.perf_counter() - start_time
            self.compilation_times[model_name] = compilation_time
            
            self.logger.info(f"Successfully compiled {model_name} in {compilation_time:.3f}s")
            return compiled_model
            
        except Exception as e:
            self.logger.warning(f"Failed to compile {model_name}: {e}")
            self.compilation_failures.add(model_name)
            
            if self.fallback_on_failure:
                return None
            else:
                raise
    
    def compile_function(
        self, 
        func: Callable, 
        example_inputs: Union[torch.Tensor, tuple], 
        func_name: str
    ) -> Optional[torch.jit.ScriptFunction]:
        """
        Compile a function with JIT optimization.
        
        Args:
            func: Function to compile
            example_inputs: Example inputs for compilation
            func_name: Name for tracking and caching
            
        Returns:
            Compiled function or None if compilation failed
        """
        if not self.enable_jit or func_name in self.compilation_failures:
            return None
        
        if func_name in self.compiled_models:
            return self.compiled_models[func_name]
        
        try:
            self.logger.info(f"Compiling function: {func_name}")
            start_time = time.perf_counter()
            
            # Compile function
            if isinstance(example_inputs, (list, tuple)):
                compiled_func = torch.jit.trace(func, example_inputs)
            else:
                compiled_func = torch.jit.trace(func, (example_inputs,))
            
            # Warm up
            self._warmup_compiled_function(compiled_func, example_inputs)
            
            # Cache
            self.compiled_models[func_name] = compiled_func
            
            compilation_time = time.perf_counter() - start_time
            self.compilation_times[func_name] = compilation_time
            
            self.logger.info(f"Successfully compiled function {func_name} in {compilation_time:.3f}s")
            return compiled_func
            
        except Exception as e:
            self.logger.warning(f"Failed to compile function {func_name}: {e}")
            self.compilation_failures.add(func_name)
            return None
    
    def _warmup_compiled_model(self, compiled_model: torch.jit.ScriptModule, example_inputs):
        """Warm up compiled model with multiple iterations."""
        try:
            with torch.no_grad():
                for _ in range(self.jit_warmup_iterations):
                    if isinstance(example_inputs, (list, tuple)):
                        _ = compiled_model(*example_inputs)
                    else:
                        _ = compiled_model(example_inputs)
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")
    
    def _warmup_compiled_function(self, compiled_func: torch.jit.ScriptFunction, example_inputs):
        """Warm up compiled function with multiple iterations."""
        try:
            for _ in range(self.jit_warmup_iterations):
                if isinstance(example_inputs, (list, tuple)):
                    _ = compiled_func(*example_inputs)
                else:
                    _ = compiled_func(example_inputs)
        except Exception as e:
            self.logger.warning(f"Function warmup failed: {e}")
    
    def get_compiled_model(self, model_name: str) -> Optional[torch.jit.ScriptModule]:
        """Get compiled model from cache."""
        return self.compiled_models.get(model_name)
    
    def clear_cache(self):
        """Clear compiled model cache."""
        self.compiled_models.clear()
        self.compilation_failures.clear()
        self.logger.info("JIT optimizer cache cleared")

class MemoryPoolManager:
    """
    Memory pool manager for efficient tensor allocation and reuse.
    
    Reduces memory allocation overhead by reusing tensor memory.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Memory pool settings
        self.enable_pooling = config.get('enable_memory_pooling', True)
        self.max_pool_size = config.get('max_pool_size_mb', 100)
        self.cleanup_threshold = config.get('cleanup_threshold', 0.8)
        
        # Tensor pools by shape and dtype
        self.tensor_pools = {}
        self.pool_usage = {}
        self.allocation_stats = {'hits': 0, 'misses': 0, 'cleanups': 0}
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info("Memory Pool Manager initialized")
    
    def get_tensor(
        self, 
        shape: tuple, 
        dtype: torch.dtype = torch.float32, 
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Get a tensor from the pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Device for tensor
            
        Returns:
            Tensor of requested shape and type
        """
        if not self.enable_pooling:
            return torch.zeros(shape, dtype=dtype, device=device)
        
        pool_key = (shape, dtype, device)
        
        with self.lock:
            if pool_key in self.tensor_pools and self.tensor_pools[pool_key]:
                # Reuse from pool
                tensor = self.tensor_pools[pool_key].pop()
                tensor.zero_()  # Clear data
                self.allocation_stats['hits'] += 1
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.zeros(shape, dtype=dtype, device=device)
                self.allocation_stats['misses'] += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        Return a tensor to the pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if not self.enable_pooling:
            return
        
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = str(tensor.device)
        
        pool_key = (shape, dtype, device)
        
        with self.lock:
            if pool_key not in self.tensor_pools:
                self.tensor_pools[pool_key] = []
                self.pool_usage[pool_key] = 0
            
            # Check pool size limits
            current_size_mb = self._calculate_pool_size_mb()
            tensor_size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
            
            if current_size_mb + tensor_size_mb < self.max_pool_size:
                self.tensor_pools[pool_key].append(tensor.detach())
                self.pool_usage[pool_key] += tensor_size_mb
            
            # Cleanup if needed
            if current_size_mb > self.max_pool_size * self.cleanup_threshold:
                self._cleanup_pools()
    
    def _calculate_pool_size_mb(self) -> float:
        """Calculate total pool size in MB."""
        return sum(self.pool_usage.values())
    
    def _cleanup_pools(self):
        """Clean up least recently used tensors from pools."""
        # Simple cleanup: remove half of each pool
        for pool_key in list(self.tensor_pools.keys()):
            pool = self.tensor_pools[pool_key]
            if len(pool) > 2:
                # Keep newer half
                keep_count = len(pool) // 2
                removed_tensors = pool[:-keep_count]
                self.tensor_pools[pool_key] = pool[-keep_count:]
                
                # Update usage tracking
                removed_size = sum(
                    t.numel() * t.element_size() / 1024 / 1024 
                    for t in removed_tensors
                )
                self.pool_usage[pool_key] -= removed_size
        
        self.allocation_stats['cleanups'] += 1
        self.logger.debug("Memory pool cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'total_size_mb': self._calculate_pool_size_mb(),
                'pool_count': len(self.tensor_pools),
                'allocation_stats': self.allocation_stats.copy(),
                'hit_rate': (
                    self.allocation_stats['hits'] / 
                    max(1, self.allocation_stats['hits'] + self.allocation_stats['misses'])
                )
            }
    
    def clear_pools(self):
        """Clear all memory pools."""
        with self.lock:
            self.tensor_pools.clear()
            self.pool_usage.clear()
            self.allocation_stats = {'hits': 0, 'misses': 0, 'cleanups': 0}
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Memory pools cleared")

class BatchOptimizer:
    """
    Batch processing optimizer for improving throughput.
    
    Automatically batches operations when beneficial and provides
    efficient batch processing utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Batch optimization settings
        self.enable_batching = config.get('enable_batching', True)
        self.min_batch_size = config.get('min_batch_size', 2)
        self.max_batch_size = config.get('max_batch_size', 32)
        self.batch_timeout_ms = config.get('batch_timeout_ms', 10)
        
        # Batching queues
        self.batch_queues = {}
        self.batch_processors = {}
        
        # Performance tracking
        self.batch_stats = {'batched_ops': 0, 'individual_ops': 0, 'batch_efficiency': 0.0}
        
        self.logger.info("Batch Optimizer initialized")
    
    def register_batch_processor(
        self, 
        operation_name: str, 
        batch_function: Callable, 
        should_batch_function: Optional[Callable] = None
    ):
        """
        Register a batch processor for an operation.
        
        Args:
            operation_name: Name of the operation
            batch_function: Function that processes batches
            should_batch_function: Function to determine if batching is beneficial
        """
        self.batch_processors[operation_name] = {
            'batch_function': batch_function,
            'should_batch': should_batch_function or (lambda inputs: len(inputs) >= self.min_batch_size)
        }
        
        self.logger.info(f"Registered batch processor for {operation_name}")
    
    def process_with_batching(
        self, 
        operation_name: str, 
        inputs: List[Any], 
        individual_function: Callable
    ) -> List[Any]:
        """
        Process inputs with automatic batching optimization.
        
        Args:
            operation_name: Name of the operation
            inputs: List of inputs to process
            individual_function: Function for individual processing (fallback)
            
        Returns:
            List of results
        """
        if not self.enable_batching or operation_name not in self.batch_processors:
            # Process individually
            results = [individual_function(inp) for inp in inputs]
            self.batch_stats['individual_ops'] += len(inputs)
            return results
        
        processor = self.batch_processors[operation_name]
        
        # Check if batching is beneficial
        if processor['should_batch'](inputs):
            # Process as batch
            try:
                results = processor['batch_function'](inputs)
                self.batch_stats['batched_ops'] += len(inputs)
                
                # Update efficiency
                if self.batch_stats['batched_ops'] + self.batch_stats['individual_ops'] > 0:
                    self.batch_stats['batch_efficiency'] = (
                        self.batch_stats['batched_ops'] / 
                        (self.batch_stats['batched_ops'] + self.batch_stats['individual_ops'])
                    )
                
                return results
                
            except Exception as e:
                self.logger.warning(f"Batch processing failed for {operation_name}: {e}")
                # Fallback to individual processing
                results = [individual_function(inp) for inp in inputs]
                self.batch_stats['individual_ops'] += len(inputs)
                return results
        else:
            # Process individually
            results = [individual_function(inp) for inp in inputs]
            self.batch_stats['individual_ops'] += len(inputs)
            return results
    
    def create_optimal_batches(
        self, 
        inputs: List[Any], 
        max_batch_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Create optimal batches from inputs.
        
        Args:
            inputs: List of inputs to batch
            max_batch_size: Maximum batch size (uses config default if None)
            
        Returns:
            List of batches
        """
        if not inputs:
            return []
        
        batch_size = max_batch_size or self.max_batch_size
        batches = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.batch_stats.copy()

class CacheOptimizer:
    """
    Advanced caching optimizer with intelligent cache management.
    
    Provides multi-level caching with automatic eviction and
    performance-based cache optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache settings
        self.enable_caching = config.get('enable_caching', True)
        self.l1_cache_size = config.get('l1_cache_size', 100)
        self.l2_cache_size = config.get('l2_cache_size', 1000)
        self.cache_ttl_seconds = config.get('cache_ttl_seconds', 60)
        
        # Multi-level caches
        self.l1_cache = {}  # Fastest access, smallest size
        self.l2_cache = {}  # Larger, slightly slower
        
        # Cache metadata
        self.access_counts = {}
        self.access_times = {}
        self.cache_stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info("Cache Optimizer initialized")
    
    @lru_cache(maxsize=1000)
    def _create_cache_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        import hashlib
        
        # Convert args and kwargs to string
        key_data = str(args) + str(sorted(kwargs.items()))
        
        # Hash for consistent key
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> tuple[Any, bool]:
        """
        Get result from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Tuple of (result, cache_hit)
        """
        if not self.enable_caching:
            return None, False
        
        current_time = time.time()
        
        with self.lock:
            # Check L1 cache first
            if cache_key in self.l1_cache:
                result, timestamp = self.l1_cache[cache_key]
                if current_time - timestamp < self.cache_ttl_seconds:
                    self.cache_stats['l1_hits'] += 1
                    self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                    self.access_times[cache_key] = current_time
                    return result, True
                else:
                    # Expired, remove from L1
                    del self.l1_cache[cache_key]
            
            # Check L2 cache
            if cache_key in self.l2_cache:
                result, timestamp = self.l2_cache[cache_key]
                if current_time - timestamp < self.cache_ttl_seconds:
                    self.cache_stats['l2_hits'] += 1
                    self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                    self.access_times[cache_key] = current_time
                    
                    # Promote to L1 if frequently accessed
                    if self.access_counts[cache_key] > 3:
                        self._promote_to_l1(cache_key, result, timestamp)
                    
                    return result, True
                else:
                    # Expired, remove from L2
                    del self.l2_cache[cache_key]
            
            # Cache miss
            self.cache_stats['misses'] += 1
            return None, False
    
    def cache_result(self, cache_key: str, result: Any):
        """
        Cache a result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.enable_caching:
            return
        
        current_time = time.time()
        
        with self.lock:
            # Store in L2 cache initially
            self.l2_cache[cache_key] = (result, current_time)
            
            # Manage L2 cache size
            if len(self.l2_cache) > self.l2_cache_size:
                self._evict_from_l2()
            
            # Initialize access tracking
            self.access_counts[cache_key] = 1
            self.access_times[cache_key] = current_time
    
    def _promote_to_l1(self, cache_key: str, result: Any, timestamp: float):
        """Promote frequently accessed item to L1 cache."""
        self.l1_cache[cache_key] = (result, timestamp)
        
        # Manage L1 cache size
        if len(self.l1_cache) > self.l1_cache_size:
            self._evict_from_l1()
    
    def _evict_from_l1(self):
        """Evict least recently used item from L1 cache."""
        if not self.l1_cache:
            return
        
        # Find LRU item
        lru_key = min(
            self.l1_cache.keys(),
            key=lambda k: self.access_times.get(k, 0)
        )
        
        # Move back to L2
        result, timestamp = self.l1_cache[lru_key]
        self.l2_cache[lru_key] = (result, timestamp)
        
        del self.l1_cache[lru_key]
    
    def _evict_from_l2(self):
        """Evict least recently used items from L2 cache."""
        if len(self.l2_cache) <= self.l2_cache_size:
            return
        
        # Remove 10% of least recently used items
        num_to_remove = max(1, len(self.l2_cache) // 10)
        
        lru_keys = sorted(
            self.l2_cache.keys(),
            key=lambda k: self.access_times.get(k, 0)
        )[:num_to_remove]
        
        for key in lru_keys:
            del self.l2_cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(self.cache_stats.values())
            hit_rate = (
                (self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']) /
                max(1, total_accesses)
            )
            
            return {
                'l1_size': len(self.l1_cache),
                'l2_size': len(self.l2_cache),
                'hit_rate': hit_rate,
                'l1_hit_rate': self.cache_stats['l1_hits'] / max(1, total_accesses),
                'l2_hit_rate': self.cache_stats['l2_hits'] / max(1, total_accesses),
                'cache_stats': self.cache_stats.copy()
            }
    
    def clear_cache(self):
        """Clear all caches."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.cache_stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}
        
        self.logger.info("All caches cleared")

class PerformanceOptimizationEngine:
    """
    Main performance optimization engine that coordinates all optimizations.
    
    Integrates JIT compilation, memory pooling, batch optimization,
    and caching for maximum performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize optimization components
        self.jit_optimizer = JITOptimizer(config.get('jit', {}))
        self.memory_pool = MemoryPoolManager(config.get('memory_pool', {}))
        self.batch_optimizer = BatchOptimizer(config.get('batch', {}))
        self.cache_optimizer = CacheOptimizer(config.get('cache', {}))
        
        # Global optimization settings
        self.optimization_level = config.get('optimization_level', 'aggressive')  # conservative, balanced, aggressive
        self.enable_profiling = config.get('enable_profiling', True)
        
        # Performance tracking
        self.optimization_stats = {
            'optimizations_applied': 0,
            'performance_improvements': [],
            'total_time_saved_ms': 0.0
        }
        
        self.logger.info(f"Performance Optimization Engine initialized (level: {self.optimization_level})")
    
    def optimize_intelligence_hub(self, intelligence_hub) -> Dict[str, Any]:
        """
        Apply comprehensive optimizations to intelligence hub.
        
        Args:
            intelligence_hub: IntelligenceHub instance to optimize
            
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {}
        
        try:
            # 1. JIT compile gating network
            if hasattr(intelligence_hub, 'gating_network'):
                example_context = torch.randn(1, 6)
                compiled_gating = self.jit_optimizer.compile_model(
                    intelligence_hub.gating_network,
                    example_context,
                    'gating_network'
                )
                
                if compiled_gating is not None:
                    intelligence_hub.gating_network_jit = compiled_gating
                    optimization_results['gating_jit'] = 'success'
                else:
                    optimization_results['gating_jit'] = 'failed'
            
            # 2. Optimize attention mechanisms
            if hasattr(intelligence_hub, 'attention_optimizer'):
                self._optimize_attention_system(intelligence_hub.attention_optimizer)
                optimization_results['attention_optimization'] = 'applied'
            
            # 3. Set up caching for regime detection
            if hasattr(intelligence_hub, 'regime_detector'):
                self._optimize_regime_detection(intelligence_hub.regime_detector)
                optimization_results['regime_caching'] = 'applied'
            
            # 4. Memory optimization
            self._apply_memory_optimizations(intelligence_hub)
            optimization_results['memory_optimization'] = 'applied'
            
            self.optimization_stats['optimizations_applied'] += 1
            
            self.logger.info("Intelligence Hub optimization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error optimizing intelligence hub: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _optimize_attention_system(self, attention_optimizer):
        """Optimize attention computation system."""
        
        # Pre-compile common attention patterns
        common_shapes = [(1, 7), (4, 7), (8, 7)]
        
        for shape in common_shapes:
            example_tensor = torch.randn(shape)
            
            # JIT compile attention function
            def attention_func(x):
                return torch.softmax(x, dim=-1)
            
            compiled_attention = self.jit_optimizer.compile_function(
                attention_func,
                example_tensor,
                f'attention_{shape[0]}x{shape[1]}'
            )
            
            if compiled_attention is not None:
                # Store compiled function for later use
                if not hasattr(attention_optimizer, 'compiled_functions'):
                    attention_optimizer.compiled_functions = {}
                attention_optimizer.compiled_functions[shape] = compiled_attention
    
    def _optimize_regime_detection(self, regime_detector):
        """Optimize regime detection with advanced caching."""
        
        # Enhance caching with performance optimizer cache
        original_detect = regime_detector.detect_regime
        
        def cached_detect_regime(market_context):
            cache_key = self.cache_optimizer._create_cache_key(**market_context)
            
            # Try cache first
            cached_result, cache_hit = self.cache_optimizer.get_cached_result(cache_key)
            if cache_hit:
                return cached_result
            
            # Compute and cache
            result = original_detect(market_context)
            self.cache_optimizer.cache_result(cache_key, result)
            
            return result
        
        # Replace with cached version
        regime_detector.detect_regime = cached_detect_regime
    
    def _apply_memory_optimizations(self, intelligence_hub):
        """Apply memory optimizations to intelligence hub."""
        
        # Enable tensor reuse in attention computations
        if hasattr(intelligence_hub, 'attention_optimizer'):
            attention_optimizer = intelligence_hub.attention_optimizer
            
            # Add memory pool to attention optimizer
            attention_optimizer.memory_pool = self.memory_pool
            
            # Override tensor creation methods to use pooling
            original_get_tensor = getattr(attention_optimizer, '_get_tensor', None)
            
            def pooled_get_tensor(shape, dtype=torch.float32, device='cpu'):
                return self.memory_pool.get_tensor(shape, dtype, device)
            
            attention_optimizer._get_tensor = pooled_get_tensor
    
    def profile_intelligence_pipeline(
        self, 
        intelligence_hub, 
        market_context: Dict[str, Any],
        agent_predictions: List[Dict[str, Any]],
        attention_weights: List[torch.Tensor],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        \"\"\"
        Profile intelligence pipeline performance.
        
        Args:
            intelligence_hub: IntelligenceHub instance
            market_context: Sample market context
            agent_predictions: Sample agent predictions
            attention_weights: Sample attention weights
            num_iterations: Number of profiling iterations
            
        Returns:
            Profiling results
        """
        if not self.enable_profiling:
            return {'profiling': 'disabled'}
        
        self.logger.info(\"Starting intelligence pipeline profiling\")
        
        # Warm up
        for _ in range(10):
            intelligence_hub.process_intelligence_pipeline(
                market_context, agent_predictions, attention_weights
            )
        
        # Profile execution
        execution_times = []
        component_times = {'regime': [], 'gating': [], 'attention': [], 'integration': []}
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            # Profile regime detection
            regime_start = time.perf_counter()
            regime_analysis = intelligence_hub._fast_regime_detection(market_context)
            regime_time = (time.perf_counter() - regime_start) * 1000
            component_times['regime'].append(regime_time)
            
            # Profile gating
            gating_start = time.perf_counter()
            gating_weights = intelligence_hub._fast_gating_computation(market_context, regime_analysis)
            gating_time = (time.perf_counter() - gating_start) * 1000
            component_times['gating'].append(gating_time)
            
            # Profile attention
            attention_start = time.perf_counter()
            attention_analysis = intelligence_hub._analyze_attention_patterns(attention_weights, regime_analysis)
            attention_time = (time.perf_counter() - attention_start) * 1000
            component_times['attention'].append(attention_time)
            
            # Profile integration
            integration_start = time.perf_counter()
            integrated_result = intelligence_hub._integrate_intelligence_components(
                regime_analysis, gating_weights, attention_analysis, agent_predictions
            )
            integration_time = (time.perf_counter() - integration_start) * 1000
            component_times['integration'].append(integration_time)
            
            total_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(total_time)
        
        # Calculate statistics
        profiling_results = {
            'total_execution': {
                'mean_ms': np.mean(execution_times),
                'p50_ms': np.percentile(execution_times, 50),
                'p95_ms': np.percentile(execution_times, 95),
                'p99_ms': np.percentile(execution_times, 99),
                'std_ms': np.std(execution_times)
            },
            'component_breakdown': {}
        }
        
        for component, times in component_times.items():
            profiling_results['component_breakdown'][component] = {
                'mean_ms': np.mean(times),
                'p95_ms': np.percentile(times, 95),
                'std_ms': np.std(times),
                'percentage_of_total': np.mean(times) / np.mean(execution_times) * 100
            }
        
        # Performance recommendations
        recommendations = self._generate_performance_recommendations(profiling_results)
        profiling_results['recommendations'] = recommendations
        
        self.logger.info(\"Intelligence pipeline profiling completed\")
        return profiling_results
    
    def _generate_performance_recommendations(self, profiling_results: Dict[str, Any]) -> List[str]:
        \"\"\"Generate performance optimization recommendations based on profiling.\"\"\"
        
        recommendations = []
        total_mean = profiling_results['total_execution']['mean_ms']
        
        # Check if total time is acceptable
        if total_mean > 5.0:
            recommendations.append(f\"Total execution time {total_mean:.2f}ms exceeds 5ms target - optimization needed\")
        
        # Check component balance
        component_breakdown = profiling_results['component_breakdown']
        
        for component, stats in component_breakdown.items():
            percentage = stats['percentage_of_total']
            
            if percentage > 40:
                recommendations.append(
                    f\"Component '{component}' consumes {percentage:.1f}% of execution time - consider optimization\"
                )
            
            if stats['std_ms'] > stats['mean_ms'] * 0.5:
                recommendations.append(
                    f\"Component '{component}' has high variance - consider stability improvements\"
                )
        
        # Optimization level specific recommendations
        if self.optimization_level == 'aggressive':
            if total_mean > 2.0:
                recommendations.append(\"Consider more aggressive JIT compilation or memory optimization\")
            
            cache_stats = self.cache_optimizer.get_cache_statistics()
            if cache_stats.get('hit_rate', 0) < 0.7:
                recommendations.append(\"Cache hit rate is low - consider tuning cache parameters\")
        
        return recommendations
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive optimization statistics.\"\"\"
        
        return {
            'optimization_stats': self.optimization_stats,
            'jit_stats': {
                'compiled_models': len(self.jit_optimizer.compiled_models),
                'compilation_failures': len(self.jit_optimizer.compilation_failures),
                'compilation_times': self.jit_optimizer.compilation_times
            },
            'memory_stats': self.memory_pool.get_statistics(),
            'batch_stats': self.batch_optimizer.get_batch_statistics(),
            'cache_stats': self.cache_optimizer.get_cache_statistics()
        }
    
    def reset_optimization_state(self):
        \"\"\"Reset all optimization state.\"\"\"
        
        self.jit_optimizer.clear_cache()
        self.memory_pool.clear_pools()
        self.cache_optimizer.clear_cache()
        
        self.optimization_stats = {
            'optimizations_applied': 0,
            'performance_improvements': [],
            'total_time_saved_ms': 0.0
        }
        
        self.logger.info(\"Optimization state reset completed\")