#!/usr/bin/env python3
"""
Integrated Performance Engine
Combines all performance optimization components into a unified system
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, asynccontextmanager
import psutil
import gc
from functools import wraps
from enum import Enum

# Import our optimization components
from .advanced_caching_system import MultiLevelCache, get_global_cache
from .jit_optimized_engine import JITModelWrapper, OptimizationConfig, OptimizedModelFactory
from .async_processing_engine import AsyncWorkflowEngine, AsyncModelInference, AsyncModelPipeline
from .memory_optimization_system import MemoryOptimizationManager, get_global_memory_manager
from .config_tuning_system import ConfigurationManager, OptimizationGoal, PerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class OptimizationProfile:
    """Performance optimization profile"""
    name: str
    level: PerformanceLevel
    enable_jit: bool = True
    enable_quantization: bool = False
    enable_async: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_config_tuning: bool = True
    target_latency_ms: float = 10.0
    target_throughput_qps: float = 100.0
    max_memory_mb: float = 1000.0
    description: str = ""

class IntegratedPerformanceEngine:
    """Main integrated performance optimization engine"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.optimization_profiles = self._create_default_profiles()
        self.current_profile = self.optimization_profiles['standard']
        
        # Core components
        self.cache = get_global_cache()
        self.memory_manager = get_global_memory_manager()
        self.config_manager = ConfigurationManager(config_file)
        self.workflow_engine = None
        
        # Model management
        self.optimized_models = {}
        self.model_pipelines = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_optimization_time': 0.0,
            'memory_saved_mb': 0.0,
            'latency_improvements': []
        }
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.optimization_lock = threading.RLock()
        
        logger.info("Integrated Performance Engine initialized")
    
    def _create_default_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create default optimization profiles"""
        profiles = {
            'basic': OptimizationProfile(
                name='basic',
                level=PerformanceLevel.BASIC,
                enable_jit=True,
                enable_quantization=False,
                enable_async=False,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_config_tuning=False,
                target_latency_ms=50.0,
                target_throughput_qps=50.0,
                max_memory_mb=500.0,
                description="Basic optimizations for development"
            ),
            'standard': OptimizationProfile(
                name='standard',
                level=PerformanceLevel.STANDARD,
                enable_jit=True,
                enable_quantization=True,
                enable_async=True,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_config_tuning=True,
                target_latency_ms=10.0,
                target_throughput_qps=200.0,
                max_memory_mb=1000.0,
                description="Standard optimizations for production"
            ),
            'aggressive': OptimizationProfile(
                name='aggressive',
                level=PerformanceLevel.AGGRESSIVE,
                enable_jit=True,
                enable_quantization=True,
                enable_async=True,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_config_tuning=True,
                target_latency_ms=5.0,
                target_throughput_qps=500.0,
                max_memory_mb=2000.0,
                description="Aggressive optimizations for high performance"
            ),
            'maximum': OptimizationProfile(
                name='maximum',
                level=PerformanceLevel.MAXIMUM,
                enable_jit=True,
                enable_quantization=True,
                enable_async=True,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_config_tuning=True,
                target_latency_ms=1.0,
                target_throughput_qps=1000.0,
                max_memory_mb=4000.0,
                description="Maximum optimizations for extreme performance"
            )
        }
        return profiles
    
    async def initialize(self, profile_name: str = 'standard'):
        """Initialize the performance engine"""
        with self.optimization_lock:
            if self.is_initialized:
                return
            
            try:
                # Set optimization profile
                self.set_profile(profile_name)
                
                # Initialize components based on profile
                if self.current_profile.enable_async:
                    self.workflow_engine = AsyncWorkflowEngine(max_workers=8)
                    await self.workflow_engine.start()
                
                # Start adaptive configuration tuning
                if self.current_profile.enable_config_tuning:
                    self.config_manager.start_adaptive_tuning(self._evaluate_performance)
                
                # Start performance monitoring
                self.performance_monitor.start_monitoring()
                
                self.is_initialized = True
                self.is_running = True
                
                logger.info(f"Performance engine initialized with profile: {profile_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize performance engine: {e}")
                raise
    
    async def shutdown(self):
        """Shutdown the performance engine"""
        with self.optimization_lock:
            if not self.is_running:
                return
            
            try:
                # Stop components
                if self.workflow_engine:
                    await self.workflow_engine.stop()
                
                self.config_manager.stop_adaptive_tuning()
                self.performance_monitor.stop_monitoring()
                
                # Cleanup resources
                self.memory_manager.cleanup()
                
                self.is_running = False
                
                logger.info("Performance engine shutdown completed")
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    def set_profile(self, profile_name: str):
        """Set optimization profile"""
        if profile_name not in self.optimization_profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        self.current_profile = self.optimization_profiles[profile_name]
        logger.info(f"Set optimization profile to: {profile_name}")
    
    def optimize_model(self, model: nn.Module, model_name: str, 
                      example_input: torch.Tensor = None) -> JITModelWrapper:
        """Optimize a model with current profile settings"""
        with self.optimization_lock:
            start_time = time.time()
            
            try:
                # Create optimization config based on profile
                config = OptimizationConfig(
                    enable_jit=self.current_profile.enable_jit,
                    enable_quantization=self.current_profile.enable_quantization,
                    enable_vectorization=True,
                    enable_async=self.current_profile.enable_async,
                    target_latency_ms=self.current_profile.target_latency_ms
                )
                
                # Optimize model
                optimized_model = JITModelWrapper(model, config)
                
                # Warm up model
                if example_input is not None:
                    optimized_model.warmup(100)
                
                # Store optimized model
                self.optimized_models[model_name] = optimized_model
                
                # Update statistics
                optimization_time = time.time() - start_time
                self.optimization_stats['total_optimizations'] += 1
                self.optimization_stats['successful_optimizations'] += 1
                self.optimization_stats['total_optimization_time'] += optimization_time
                
                logger.info(f"Model {model_name} optimized in {optimization_time:.2f}s")
                
                return optimized_model
                
            except Exception as e:
                self.optimization_stats['failed_optimizations'] += 1
                logger.error(f"Failed to optimize model {model_name}: {e}")
                raise
    
    async def create_model_pipeline(self, models: Dict[str, nn.Module], 
                                   pipeline_name: str,
                                   preprocessing: Optional[Callable] = None,
                                   postprocessing: Optional[Callable] = None) -> AsyncModelPipeline:
        """Create optimized model pipeline"""
        try:
            # Optimize all models in the pipeline
            optimized_models = {}
            for name, model in models.items():
                optimized_models[name] = self.optimize_model(model, f"{pipeline_name}_{name}")
            
            # Create async pipeline
            pipeline = AsyncModelPipeline(
                models=optimized_models,
                preprocessing=preprocessing,
                postprocessing=postprocessing
            )
            
            await pipeline.start()
            
            self.model_pipelines[pipeline_name] = pipeline
            
            logger.info(f"Model pipeline {pipeline_name} created with {len(models)} models")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create model pipeline {pipeline_name}: {e}")
            raise
    
    async def batch_inference(self, model_name: str, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform batch inference with optimizations"""
        if model_name not in self.optimized_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.optimized_models[model_name]
        
        # Use async inference if available
        if self.current_profile.enable_async and hasattr(model, 'async_inference'):
            return await model.async_inference(inputs)
        else:
            # Standard batch inference
            batch_tensor = torch.stack(inputs)
            
            with self.performance_monitor.measure_inference(model_name):
                result = model(batch_tensor)
            
            return [result[i] for i in range(len(inputs))]
    
    def _evaluate_performance(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Evaluate performance for configuration tuning"""
        # This would be implemented based on specific models and workloads
        # For now, return mock metrics
        return PerformanceMetrics(
            latency_ms=5.0,
            throughput_qps=200.0,
            memory_mb=800.0,
            cpu_usage=60.0,
            accuracy=0.95,
            stability_score=0.9
        )
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current performance"""
        recommendations = []
        
        # Analyze current performance
        current_metrics = self.performance_monitor.get_current_metrics()
        
        if current_metrics:
            # Latency recommendations
            if current_metrics.latency_ms > self.current_profile.target_latency_ms:
                recommendations.append({
                    'type': 'latency',
                    'issue': f'Latency {current_metrics.latency_ms:.1f}ms exceeds target {self.current_profile.target_latency_ms:.1f}ms',
                    'suggestions': [
                        'Enable more aggressive JIT optimization',
                        'Increase batch size',
                        'Enable quantization',
                        'Use async processing'
                    ]
                })
            
            # Throughput recommendations
            if current_metrics.throughput_qps < self.current_profile.target_throughput_qps:
                recommendations.append({
                    'type': 'throughput',
                    'issue': f'Throughput {current_metrics.throughput_qps:.1f}qps below target {self.current_profile.target_throughput_qps:.1f}qps',
                    'suggestions': [
                        'Increase number of workers',
                        'Optimize batch processing',
                        'Enable parallel inference',
                        'Use more aggressive caching'
                    ]
                })
            
            # Memory recommendations
            if current_metrics.memory_mb > self.current_profile.max_memory_mb:
                recommendations.append({
                    'type': 'memory',
                    'issue': f'Memory usage {current_metrics.memory_mb:.1f}MB exceeds limit {self.current_profile.max_memory_mb:.1f}MB',
                    'suggestions': [
                        'Enable memory pooling',
                        'Reduce cache size',
                        'Enable model quantization',
                        'Optimize garbage collection'
                    ]
                })
        
        return {
            'recommendations': recommendations,
            'current_profile': self.current_profile.name,
            'suggested_profile': self._suggest_profile(current_metrics) if current_metrics else None
        }
    
    def _suggest_profile(self, metrics: PerformanceMetrics) -> str:
        """Suggest optimal profile based on current metrics"""
        # Simple heuristic-based profile suggestion
        if metrics.latency_ms > 50:
            return 'basic'
        elif metrics.latency_ms > 10:
            return 'standard'
        elif metrics.latency_ms > 5:
            return 'aggressive'
        else:
            return 'maximum'
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'engine_stats': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'current_profile': self.current_profile.name,
                'optimized_models': len(self.optimized_models),
                'model_pipelines': len(self.model_pipelines),
                **self.optimization_stats
            },
            'cache_stats': self.cache.get_stats(),
            'memory_stats': self.memory_manager.get_comprehensive_stats(),
            'config_stats': self.config_manager.get_comprehensive_stats(),
            'performance_stats': self.performance_monitor.get_stats()
        }
        
        return stats
    
    def save_optimization_report(self, filepath: Path):
        """Save comprehensive optimization report"""
        report = {
            'timestamp': time.time(),
            'profile': self.current_profile.__dict__,
            'stats': self.get_comprehensive_stats(),
            'recommendations': self.get_optimization_recommendations(),
            'model_performance': {
                name: model.get_stats() 
                for name, model in self.optimized_models.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {filepath}")

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance tracking
        self.inference_times = defaultdict(list)
        self.throughput_samples = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_inferences': 0,
            'total_monitoring_time': 0.0,
            'performance_alerts': 0
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_worker(self):
        """Background monitoring worker"""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                # Create metrics
                metrics = PerformanceMetrics(
                    latency_ms=np.mean(list(self.inference_times.values())[-10:]) if self.inference_times else 0.0,
                    throughput_qps=np.mean(self.throughput_samples) if self.throughput_samples else 0.0,
                    memory_mb=memory_info.used / 1024 / 1024,
                    cpu_usage=cpu_percent,
                    accuracy=0.95,  # Would be measured from actual performance
                    stability_score=0.9  # Would be calculated from variance
                )
                
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
        
        self.stats['total_monitoring_time'] = time.time() - start_time
    
    @contextmanager
    def measure_inference(self, model_name: str):
        """Context manager for measuring inference time"""
        start_time = time.time()
        try:
            yield
        finally:
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times[model_name].append(inference_time)
            
            # Keep only last 100 measurements per model
            if len(self.inference_times[model_name]) > 100:
                self.inference_times[model_name].pop(0)
            
            self.stats['total_inferences'] += 1
            
            # Update throughput
            self.throughput_samples.append(1.0 / (inference_time / 1000.0))  # QPS
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            'metrics_history_length': len(self.metrics_history),
            'tracked_models': len(self.inference_times),
            'monitoring_active': self.monitoring_active
        }

# Context managers for performance optimization
@contextmanager
def performance_context(engine: IntegratedPerformanceEngine):
    """Context manager for performance optimization"""
    try:
        yield engine
    finally:
        # Trigger optimization if needed
        with engine.memory_manager.memory_optimized_context():
            pass

@asynccontextmanager
async def async_performance_context(engine: IntegratedPerformanceEngine):
    """Async context manager for performance optimization"""
    try:
        yield engine
    finally:
        # Trigger async optimization if needed
        if engine.workflow_engine:
            await engine.workflow_engine.submit_task(
                "cleanup_task",
                engine.memory_manager.optimize_memory
            )

# Decorators for performance optimization
def optimize_performance(engine: IntegratedPerformanceEngine):
    """Decorator for performance optimization"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with performance_context(engine):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_optimize_performance(engine: IntegratedPerformanceEngine):
    """Decorator for async performance optimization"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with async_performance_context(engine):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
async def main():
    """Example usage of integrated performance engine"""
    # Create performance engine
    engine = IntegratedPerformanceEngine()
    
    try:
        # Initialize with standard profile
        await engine.initialize('standard')
        
        # Create example model
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # Optimize model
        example_input = torch.randn(1, 100)
        optimized_model = engine.optimize_model(model, "example_model", example_input)
        
        # Test inference
        test_inputs = [torch.randn(100) for _ in range(10)]
        results = await engine.batch_inference("example_model", test_inputs)
        
        print(f"Inference results: {len(results)} outputs")
        
        # Get optimization recommendations
        recommendations = engine.get_optimization_recommendations()
        print(f"Recommendations: {recommendations}")
        
        # Get comprehensive stats
        stats = engine.get_comprehensive_stats()
        print(f"Engine stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Save optimization report
        engine.save_optimization_report(Path("optimization_report.json"))
        
    finally:
        # Shutdown engine
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())