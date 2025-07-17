#!/usr/bin/env python3
"""
JIT Model Preloading and Warming System
AGENT 2: Trading Engine RTO Specialist

This module implements aggressive model preloading and warming to reduce
trading engine startup time from 7.8s to <5s. 

Key Features:
- JIT model precompilation and caching
- Parallel model loading with thread pools
- Model warming with synthetic data
- Memory-mapped model loading for speed
- Startup performance monitoring
- Automatic model optimization detection
"""

import os
import sys
import time
import asyncio
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import pickle
import hashlib
import logging

import torch
import torch.jit
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, Event, EventType
from src.core.performance.performance_monitor import PerformanceMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model preloading"""
    name: str
    path: str
    warmup_input_shape: List[int]
    warmup_iterations: int = 10
    critical: bool = True
    precompile: bool = True
    memory_map: bool = True
    parallel_load: bool = True
    
@dataclass
class ModelLoadingStats:
    """Statistics for model loading performance"""
    model_name: str
    load_time_ms: float
    warmup_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    jit_compilation_time_ms: float
    optimization_applied: bool
    cache_hit: bool
    thread_id: int
    
@dataclass
class PreloadingResults:
    """Results from model preloading operation"""
    total_models: int
    loaded_models: int
    failed_models: int
    total_time_ms: float
    parallel_speedup: float
    memory_usage_mb: float
    model_stats: List[ModelLoadingStats] = field(default_factory=list)
    critical_models_loaded: bool = True
    startup_ready: bool = False


class ModelCache:
    """High-performance model caching system"""
    
    def __init__(self, cache_dir: str = "/tmp/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache_index = {}
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                self._cache_index = json.load(f)
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_path = self.cache_dir / "cache_index.json"
        with open(index_path, 'w') as f:
            json.dump(self._cache_index, f, indent=2)
    
    def get_cache_key(self, model_path: str) -> str:
        """Generate cache key for model"""
        model_stat = os.stat(model_path)
        content = f"{model_path}:{model_stat.st_mtime}:{model_stat.st_size}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_cached(self, model_path: str) -> bool:
        """Check if model is cached"""
        cache_key = self.get_cache_key(model_path)
        return cache_key in self._cache_index
    
    def get_cached_model(self, model_path: str) -> Optional[torch.jit.ScriptModule]:
        """Get cached model if available"""
        cache_key = self.get_cache_key(model_path)
        if cache_key in self._cache_index:
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                try:
                    return torch.jit.load(str(cache_path))
                except Exception as e:
                    logger.warning(f"Failed to load cached model {cache_path}: {e}")
                    # Remove invalid cache entry
                    del self._cache_index[cache_key]
                    cache_path.unlink(missing_ok=True)
        return None
    
    def cache_model(self, model_path: str, model: torch.jit.ScriptModule):
        """Cache compiled model"""
        cache_key = self.get_cache_key(model_path)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        try:
            model.save(str(cache_path))
            self._cache_index[cache_key] = {
                'model_path': model_path,
                'cache_path': str(cache_path),
                'cached_at': time.time()
            }
            self._save_cache_index()
            logger.info(f"Cached model {model_path} as {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache model {model_path}: {e}")


class JITModelPreloader:
    """
    High-performance JIT model preloader for trading engine
    
    Implements aggressive preloading strategies:
    - Parallel model loading using thread pools
    - JIT compilation with optimization
    - Model warming with synthetic data
    - Memory-mapped loading for speed
    - Intelligent caching system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = EventBus()
        self.performance_monitor = PerformanceMonitor()
        self.model_cache = ModelCache()
        
        # Threading configuration
        self.max_workers = config.get('max_workers', min(multiprocessing.cpu_count(), 8))
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Model registry
        self.models: Dict[str, torch.jit.ScriptModule] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.loading_stats: Dict[str, ModelLoadingStats] = {}
        
        # Performance tracking
        self.start_time = None
        self.total_load_time = 0.0
        self.memory_usage_start = 0
        
        # Optimization flags
        self.use_memory_mapping = config.get('use_memory_mapping', True)
        self.enable_jit_optimization = config.get('enable_jit_optimization', True)
        self.parallel_loading = config.get('parallel_loading', True)
        self.warmup_enabled = config.get('warmup_enabled', True)
        
        logger.info(f"JIT Model Preloader initialized with {self.max_workers} workers")
    
    def register_model(self, model_config: ModelConfig):
        """Register a model for preloading"""
        self.model_configs[model_config.name] = model_config
        logger.info(f"Registered model {model_config.name} for preloading")
    
    def register_models_from_config(self, models_config: List[Dict[str, Any]]):
        """Register multiple models from configuration"""
        for model_config in models_config:
            config = ModelConfig(**model_config)
            self.register_model(config)
    
    async def preload_all_models(self) -> PreloadingResults:
        """
        Preload all registered models with maximum performance
        
        Returns:
            PreloadingResults containing performance metrics
        """
        logger.info("Starting aggressive model preloading")
        self.start_time = time.time()
        self.memory_usage_start = self._get_memory_usage()
        
        # Separate critical and non-critical models
        critical_models = {k: v for k, v in self.model_configs.items() if v.critical}
        non_critical_models = {k: v for k, v in self.model_configs.items() if not v.critical}
        
        # Load critical models first (parallel)
        critical_results = await self._load_models_parallel(critical_models)
        
        # Load non-critical models (parallel, but lower priority)
        non_critical_results = await self._load_models_parallel(non_critical_models)
        
        # Combine results
        total_time = time.time() - self.start_time
        memory_usage = self._get_memory_usage() - self.memory_usage_start
        
        results = PreloadingResults(
            total_models=len(self.model_configs),
            loaded_models=len(critical_results) + len(non_critical_results),
            failed_models=len(self.model_configs) - len(critical_results) - len(non_critical_results),
            total_time_ms=total_time * 1000,
            parallel_speedup=self._calculate_speedup(),
            memory_usage_mb=memory_usage,
            model_stats=list(self.loading_stats.values()),
            critical_models_loaded=len(critical_results) == len(critical_models),
            startup_ready=len(critical_results) == len(critical_models)
        )
        
        # Log performance summary
        self._log_performance_summary(results)
        
        # Fire event for startup readiness
        if results.startup_ready:
            await self.event_bus.emit(Event(
                type=EventType.SYSTEM_READY,
                data={'component': 'model_preloader', 'results': results}
            ))
        
        return results
    
    async def _load_models_parallel(self, models: Dict[str, ModelConfig]) -> List[str]:
        """Load models in parallel using thread pool"""
        if not models:
            return []
        
        logger.info(f"Loading {len(models)} models in parallel")
        
        # Create tasks for parallel execution
        tasks = []
        loop = asyncio.get_event_loop()
        
        for model_name, model_config in models.items():
            task = loop.run_in_executor(
                self.thread_pool,
                self._load_single_model,
                model_name,
                model_config
            )
            tasks.append(task)
        
        # Wait for all models to load
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        loaded_models = []
        for model_name, result in zip(models.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load model {model_name}: {result}")
            else:
                loaded_models.append(model_name)
        
        return loaded_models
    
    def _load_single_model(self, model_name: str, model_config: ModelConfig) -> bool:
        """Load a single model with optimization"""
        load_start = time.time()
        thread_id = threading.get_ident()
        
        try:
            logger.info(f"Loading model {model_name} on thread {thread_id}")
            
            # Check if model exists
            if not Path(model_config.path).exists():
                raise FileNotFoundError(f"Model file not found: {model_config.path}")
            
            # Try to load from cache first
            cached_model = self.model_cache.get_cached_model(model_config.path)
            cache_hit = cached_model is not None
            
            if cached_model:
                logger.info(f"Using cached model for {model_name}")
                model = cached_model
                jit_compile_time = 0.0
            else:
                # Load and compile model
                jit_start = time.time()
                model = self._load_and_compile_model(model_config)
                jit_compile_time = (time.time() - jit_start) * 1000
                
                # Cache the compiled model
                self.model_cache.cache_model(model_config.path, model)
            
            # Store model
            self.models[model_name] = model
            
            # Warm up model if enabled
            warmup_start = time.time()
            if self.warmup_enabled:
                self._warm_up_model(model, model_config)
            warmup_time = (time.time() - warmup_start) * 1000
            
            # Calculate total time and memory
            total_time = (time.time() - load_start) * 1000
            memory_usage = self._get_model_memory_usage(model)
            
            # Store loading statistics
            self.loading_stats[model_name] = ModelLoadingStats(
                model_name=model_name,
                load_time_ms=total_time - warmup_time - jit_compile_time,
                warmup_time_ms=warmup_time,
                total_time_ms=total_time,
                memory_usage_mb=memory_usage,
                jit_compilation_time_ms=jit_compile_time,
                optimization_applied=self.enable_jit_optimization,
                cache_hit=cache_hit,
                thread_id=thread_id
            )
            
            logger.info(f"Successfully loaded model {model_name} in {total_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _load_and_compile_model(self, model_config: ModelConfig) -> torch.jit.ScriptModule:
        """Load and compile model with optimizations"""
        # Load model
        if self.use_memory_mapping:
            model = torch.jit.load(model_config.path, map_location='cpu')
        else:
            model = torch.load(model_config.path, map_location='cpu')
        
        # Convert to TorchScript if needed
        if not isinstance(model, torch.jit.ScriptModule):
            example_input = torch.randn(model_config.warmup_input_shape)
            model = torch.jit.trace(model, example_input)
        
        # Apply optimizations
        if self.enable_jit_optimization:
            model = torch.jit.optimize_for_inference(model)
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def _warm_up_model(self, model: torch.jit.ScriptModule, model_config: ModelConfig):
        """Warm up model with synthetic data"""
        try:
            # Generate synthetic input
            input_tensor = torch.randn(model_config.warmup_input_shape)
            
            # Warm up with multiple iterations
            with torch.no_grad():
                for _ in range(model_config.warmup_iterations):
                    _ = model(input_tensor)
            
            logger.debug(f"Warmed up model {model_config.name}")
            
        except Exception as e:
            logger.warning(f"Failed to warm up model {model_config.name}: {e}")
    
    def get_model(self, model_name: str) -> Optional[torch.jit.ScriptModule]:
        """Get preloaded model by name"""
        return self.models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return model_name in self.models
    
    def get_loading_stats(self, model_name: str) -> Optional[ModelLoadingStats]:
        """Get loading statistics for a model"""
        return self.loading_stats.get(model_name)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_model_memory_usage(self, model: torch.jit.ScriptModule) -> float:
        """Get memory usage of a specific model in MB"""
        try:
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / 1024 / 1024
        except Exception:
            return 0.0
    
    def _calculate_speedup(self) -> float:
        """Calculate parallel speedup factor"""
        if not self.loading_stats:
            return 1.0
        
        # Calculate sequential time (sum of all load times)
        sequential_time = sum(stat.total_time_ms for stat in self.loading_stats.values())
        
        # Calculate parallel time (actual elapsed time)
        parallel_time = (time.time() - self.start_time) * 1000
        
        return sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    def _log_performance_summary(self, results: PreloadingResults):
        """Log performance summary"""
        logger.info("=" * 60)
        logger.info("MODEL PRELOADING PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total models: {results.total_models}")
        logger.info(f"Successfully loaded: {results.loaded_models}")
        logger.info(f"Failed to load: {results.failed_models}")
        logger.info(f"Total time: {results.total_time_ms:.2f}ms")
        logger.info(f"Memory usage: {results.memory_usage_mb:.2f}MB")
        logger.info(f"Parallel speedup: {results.parallel_speedup:.2f}x")
        logger.info(f"Critical models loaded: {results.critical_models_loaded}")
        logger.info(f"Startup ready: {results.startup_ready}")
        
        # Log individual model stats
        logger.info("\nINDIVIDUAL MODEL STATISTICS:")
        logger.info("-" * 40)
        for stat in results.model_stats:
            logger.info(f"{stat.model_name}:")
            logger.info(f"  Load time: {stat.load_time_ms:.2f}ms")
            logger.info(f"  Warmup time: {stat.warmup_time_ms:.2f}ms")
            logger.info(f"  Total time: {stat.total_time_ms:.2f}ms")
            logger.info(f"  Memory: {stat.memory_usage_mb:.2f}MB")
            logger.info(f"  JIT compile: {stat.jit_compilation_time_ms:.2f}ms")
            logger.info(f"  Cache hit: {stat.cache_hit}")
            logger.info(f"  Thread ID: {stat.thread_id}")
        
        logger.info("=" * 60)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down model preloader")
        self.thread_pool.shutdown(wait=True)
        logger.info("Model preloader shutdown complete")


# Factory function
def create_model_preloader(config: Dict[str, Any]) -> JITModelPreloader:
    """Create JIT model preloader instance"""
    return JITModelPreloader(config)


# Default configuration
DEFAULT_CONFIG = {
    'max_workers': min(multiprocessing.cpu_count(), 8),
    'use_memory_mapping': True,
    'enable_jit_optimization': True,
    'parallel_loading': True,
    'warmup_enabled': True,
    'warmup_iterations': 10,
    'cache_enabled': True,
    'cache_dir': '/tmp/model_cache'
}

# Trading engine model configurations
TRADING_ENGINE_MODELS = [
    {
        'name': 'position_sizing_agent',
        'path': '/app/models/jit_optimized/position_sizing_agent_jit.pt',
        'warmup_input_shape': [1, 47],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    },
    {
        'name': 'stop_target_agent',
        'path': '/app/models/jit_optimized/stop_target_agent_jit.pt',
        'warmup_input_shape': [1, 47],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    },
    {
        'name': 'risk_monitor_agent',
        'path': '/app/models/jit_optimized/risk_monitor_agent_jit.pt',
        'warmup_input_shape': [1, 47],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    },
    {
        'name': 'portfolio_optimizer_agent',
        'path': '/app/models/jit_optimized/portfolio_optimizer_agent_jit.pt',
        'warmup_input_shape': [1, 47],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    },
    {
        'name': 'routing_agent',
        'path': '/app/models/jit_optimized/routing_agent_jit.pt',
        'warmup_input_shape': [1, 55],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    },
    {
        'name': 'centralized_critic',
        'path': '/app/models/jit_optimized/centralized_critic_jit.pt',
        'warmup_input_shape': [1, 47],
        'warmup_iterations': 10,
        'critical': True,
        'precompile': True,
        'memory_map': True,
        'parallel_load': True
    }
]


# CLI interface
async def main():
    """Main entry point for model preloading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JIT Model Preloader for Trading Engine")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--models-dir", type=str, default="/app/models/jit_optimized", 
                       help="Directory containing JIT models")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker threads")
    parser.add_argument("--warmup-iterations", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = DEFAULT_CONFIG.copy()
    if args.workers:
        config['max_workers'] = args.workers
    if args.warmup_iterations:
        config['warmup_iterations'] = args.warmup_iterations
    
    # Create preloader
    preloader = create_model_preloader(config)
    
    # Register models
    preloader.register_models_from_config(TRADING_ENGINE_MODELS)
    
    # Preload models
    results = await preloader.preload_all_models()
    
    # Check if startup is ready
    if results.startup_ready:
        logger.info("🚀 Trading engine startup ready!")
        sys.exit(0)
    else:
        logger.error("❌ Trading engine startup failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())