#!/usr/bin/env python3
"""
Memory Optimization System for Training Infrastructure
Optimizes memory usage, garbage collection, and system resources
"""

import os
import gc
import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from contextlib import contextmanager
import torch
import numpy as np
from functools import wraps
import threading
import weakref

@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    max_memory_percent: float = 85.0
    gc_threshold: float = 70.0
    monitoring_interval: float = 30.0
    enable_automatic_gc: bool = True
    enable_memory_profiling: bool = False
    cache_size_limit: int = 1024 * 1024 * 1024  # 1GB
    checkpoint_memory_limit: float = 80.0

class MemoryOptimizer:
    """Comprehensive memory optimization system"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.memory_stats = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Object tracking
        self.tracked_objects = weakref.WeakSet()
        self.memory_cache = {}
        
        # Setup initial optimizations
        self._setup_gc_optimizations()
        self._setup_memory_monitoring()
    
    def _setup_gc_optimizations(self):
        """Setup garbage collection optimizations"""
        # Increase GC thresholds for better performance
        gc.set_threshold(700, 10, 10)
        
        # Enable automatic garbage collection
        if self.config.enable_automatic_gc:
            gc.enable()
        
        self.logger.info("Garbage collection optimizations enabled")
    
    def _setup_memory_monitoring(self):
        """Setup memory monitoring"""
        self.start_memory_monitoring()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Python memory
        python_memory = {
            'objects': len(gc.get_objects()),
            'garbage': len(gc.garbage),
            'generation_counts': gc.get_count()
        }
        
        # GPU memory (if available)
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved': torch.cuda.memory_reserved(i) / (1024**3),
                    'max_allocated': torch.cuda.max_memory_allocated(i) / (1024**3),
                    'max_reserved': torch.cuda.max_memory_reserved(i) / (1024**3)
                }
        
        return {
            'system': {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_gb': system_memory.used / (1024**3),
                'percent': system_memory.percent,
                'free_gb': system_memory.free / (1024**3)
            },
            'process': {
                'rss_gb': process_memory.rss / (1024**3),
                'vms_gb': process_memory.vms / (1024**3),
                'percent': process.memory_percent(),
                'num_threads': process.num_threads()
            },
            'python': python_memory,
            'gpu': gpu_memory,
            'cache_size': len(self.memory_cache)
        }
    
    def optimize_memory(self, aggressive: bool = False):
        """Optimize memory usage"""
        initial_memory = self.get_memory_info()
        
        # Clear Python garbage
        collected = gc.collect()
        
        # Clear caches
        self.clear_cache()
        
        # PyTorch memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # NumPy memory optimization
        if hasattr(np, 'clear_cache'):
            np.clear_cache()
        
        if aggressive:
            # More aggressive optimizations
            self._aggressive_memory_cleanup()
        
        final_memory = self.get_memory_info()
        
        memory_freed = (
            initial_memory['system']['used_gb'] - 
            final_memory['system']['used_gb']
        )
        
        self.logger.info(f"Memory optimization completed. "
                        f"Freed {memory_freed:.2f}GB, "
                        f"collected {collected} objects")
        
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_freed_gb': memory_freed,
            'objects_collected': collected
        }
    
    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup"""
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear weak references
        self.tracked_objects.clear()
        
        # Clear module caches
        if hasattr(sys, 'modules'):
            for module in list(sys.modules.values()):
                if hasattr(module, '__dict__'):
                    for attr_name in list(module.__dict__.keys()):
                        if attr_name.startswith('_cache'):
                            delattr(module, attr_name)
    
    def clear_cache(self):
        """Clear memory cache"""
        cleared_items = len(self.memory_cache)
        self.memory_cache.clear()
        self.logger.info(f"Cleared {cleared_items} items from memory cache")
    
    def cache_object(self, key: str, obj: Any, size_estimate: int = 0):
        """Cache object with memory limit"""
        if len(self.memory_cache) * 1000 > self.config.cache_size_limit:
            # Remove oldest items
            keys_to_remove = list(self.memory_cache.keys())[:10]
            for key in keys_to_remove:
                del self.memory_cache[key]
        
        self.memory_cache[key] = obj
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Get cached object"""
        return self.memory_cache.get(key)
    
    def start_memory_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Memory monitoring started")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                self.memory_stats.append({
                    'timestamp': time.time(),
                    'memory_info': memory_info
                })
                
                # Keep only recent stats
                if len(self.memory_stats) > 1000:
                    self.memory_stats = self.memory_stats[-500:]
                
                # Check for memory alerts
                self._check_memory_alerts(memory_info)
                
                time.sleep(self.config.monitoring_interval)
            
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _check_memory_alerts(self, memory_info: Dict[str, Any]):
        """Check for memory alerts"""
        system_percent = memory_info['system']['percent']
        
        if system_percent > self.config.max_memory_percent:
            self.logger.warning(f"High memory usage: {system_percent:.1f}%")
            
            # Automatic garbage collection
            if self.config.enable_automatic_gc:
                gc.collect()
        
        if system_percent > self.config.gc_threshold:
            self.logger.info(f"Memory usage {system_percent:.1f}%, triggering GC")
            gc.collect()
    
    @contextmanager
    def memory_profiler(self, name: str = "operation"):
        """Context manager for memory profiling"""
        if not self.config.enable_memory_profiling:
            yield
            return
        
        start_memory = self.get_memory_info()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_info()
            
            memory_delta = (
                end_memory['system']['used_gb'] - 
                start_memory['system']['used_gb']
            )
            
            self.logger.info(f"Memory Profile [{name}]: "
                           f"Time={end_time-start_time:.2f}s, "
                           f"Memory Δ={memory_delta:.3f}GB")
    
    def memory_efficient_decorator(self, func: Callable) -> Callable:
        """Decorator for memory-efficient functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.memory_profiler(func.__name__):
                # Pre-execution optimization
                if self.get_memory_info()['system']['percent'] > self.config.gc_threshold:
                    gc.collect()
                
                result = func(*args, **kwargs)
                
                # Post-execution cleanup
                if self.get_memory_info()['system']['percent'] > self.config.gc_threshold:
                    gc.collect()
                
                return result
        
        return wrapper
    
    def optimize_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for memory-efficient training"""
        # Enable gradient checkpointing if supported
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Enabled gradient checkpointing")
        
        # Track model in weak references
        self.tracked_objects.add(model)
        
        return model
    
    def create_memory_efficient_dataloader(self, dataset, batch_size: int = 32, 
                                         num_workers: int = 4, **kwargs) -> torch.utils.data.DataLoader:
        """Create memory-efficient dataloader"""
        # Optimize number of workers based on memory
        memory_info = self.get_memory_info()
        available_memory_gb = memory_info['system']['available_gb']
        
        # Reduce workers if memory is limited
        if available_memory_gb < 8:
            num_workers = min(num_workers, 2)
        elif available_memory_gb < 16:
            num_workers = min(num_workers, 4)
        
        # Optimize batch size
        if available_memory_gb < 8:
            batch_size = min(batch_size, 16)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **kwargs
        )
    
    def checkpoint_memory_check(self) -> bool:
        """Check if memory is suitable for checkpointing"""
        memory_info = self.get_memory_info()
        memory_percent = memory_info['system']['percent']
        
        if memory_percent > self.config.checkpoint_memory_limit:
            self.logger.warning(f"Memory usage {memory_percent:.1f}% too high for checkpointing")
            return False
        
        return True
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations"""
        memory_info = self.get_memory_info()
        recommendations = []
        
        # System memory recommendations
        system_percent = memory_info['system']['percent']
        if system_percent > 80:
            recommendations.append({
                'type': 'system_memory',
                'priority': 'high',
                'message': f"System memory usage {system_percent:.1f}% is high. Consider reducing batch size or enabling gradient checkpointing."
            })
        
        # GPU memory recommendations
        if memory_info['gpu']:
            for gpu_id, gpu_info in memory_info['gpu'].items():
                allocated_gb = gpu_info['allocated']
                reserved_gb = gpu_info['reserved']
                
                if allocated_gb > 8:
                    recommendations.append({
                        'type': 'gpu_memory',
                        'priority': 'medium',
                        'message': f"{gpu_id} has {allocated_gb:.1f}GB allocated. Consider mixed precision training."
                    })
        
        # Python object recommendations
        if memory_info['python']['objects'] > 1000000:
            recommendations.append({
                'type': 'python_objects',
                'priority': 'medium',
                'message': f"High number of Python objects ({memory_info['python']['objects']}). Consider manual garbage collection."
            })
        
        return {
            'memory_info': memory_info,
            'recommendations': recommendations,
            'memory_stats_count': len(self.memory_stats)
        }
    
    def save_memory_report(self, filepath: str):
        """Save comprehensive memory report"""
        report = {
            'config': {
                'max_memory_percent': self.config.max_memory_percent,
                'gc_threshold': self.config.gc_threshold,
                'monitoring_interval': self.config.monitoring_interval,
                'enable_automatic_gc': self.config.enable_automatic_gc
            },
            'current_memory': self.get_memory_info(),
            'recommendations': self.get_memory_recommendations(),
            'memory_stats': self.memory_stats[-100:],  # Last 100 measurements
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Memory report saved to {filepath}")
    
    def cleanup(self):
        """Cleanup memory optimizer"""
        self.stop_memory_monitoring()
        self.clear_cache()
        self.tracked_objects.clear()
        
        # Final garbage collection
        gc.collect()
        
        self.logger.info("Memory optimizer cleaned up")

# Factory functions and utilities
def create_memory_optimizer(max_memory_percent: float = 85.0,
                          enable_monitoring: bool = True) -> MemoryOptimizer:
    """Create memory optimizer with default settings"""
    config = MemoryConfig(
        max_memory_percent=max_memory_percent,
        enable_automatic_gc=True,
        monitoring_interval=30.0 if enable_monitoring else 0
    )
    
    return MemoryOptimizer(config)

def memory_efficient_training_setup(model: torch.nn.Module, 
                                  batch_size: int = 32) -> tuple:
    """Setup memory-efficient training configuration"""
    optimizer = create_memory_optimizer()
    
    # Optimize model
    optimized_model = optimizer.optimize_for_training(model)
    
    # Get memory recommendations
    recommendations = optimizer.get_memory_recommendations()
    
    # Adjust batch size based on memory
    memory_info = optimizer.get_memory_info()
    available_memory_gb = memory_info['system']['available_gb']
    
    if available_memory_gb < 8:
        batch_size = min(batch_size, 16)
    elif available_memory_gb < 16:
        batch_size = min(batch_size, 32)
    
    return optimized_model, batch_size, optimizer, recommendations

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = create_memory_optimizer()
    
    # Get memory info
    memory_info = optimizer.get_memory_info()
    print(f"Memory Info: {json.dumps(memory_info, indent=2)}")
    
    # Test memory optimization
    optimization_result = optimizer.optimize_memory()
    print(f"Optimization Result: {optimization_result}")
    
    # Get recommendations
    recommendations = optimizer.get_memory_recommendations()
    print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
    
    # Test memory profiling
    with optimizer.memory_profiler("test_operation"):
        # Simulate memory-intensive operation
        data = [i for i in range(1000000)]
        del data
    
    # Save report
    optimizer.save_memory_report("/tmp/memory_report.json")
    
    # Cleanup
    optimizer.cleanup()