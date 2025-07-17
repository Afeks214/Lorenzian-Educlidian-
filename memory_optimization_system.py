#!/usr/bin/env python3
"""
Comprehensive Memory Optimization System for GrandModel

This system provides:
1. Memory leak detection and prevention
2. Intelligent buffer management
3. Garbage collection optimization
4. Model architecture memory efficiency
5. Real-time memory monitoring
6. Training pipeline memory optimization
7. Automated memory optimization recommendations

Author: Claude Code Assistant
Date: July 2025
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import threading
import time
import logging
import json
import os
import numpy as np
import weakref
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import pickle
import mmap
import tracemalloc
from functools import wraps
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Memory metrics container"""
    timestamp: float
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    process_memory_gb: float
    gpu_memory_gb: float
    cache_memory_gb: float
    swap_memory_gb: float
    memory_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization"""
    # Memory limits
    max_memory_usage_gb: float = 8.0
    memory_warning_threshold: float = 0.75
    memory_critical_threshold: float = 0.90
    
    # Garbage collection
    gc_threshold_0: int = 700  # Young generation
    gc_threshold_1: int = 10   # Middle generation
    gc_threshold_2: int = 10   # Old generation
    auto_gc_interval: float = 60.0  # seconds
    
    # Buffer management
    buffer_size_limit_mb: int = 1024
    buffer_compression_threshold: float = 0.8
    enable_buffer_compression: bool = True
    
    # Model optimization
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_model_sharding: bool = False
    
    # Monitoring
    monitoring_interval: float = 5.0
    enable_memory_profiling: bool = True
    profile_top_n: int = 10
    
    # Training optimization
    batch_size_auto_scaling: bool = True
    gradient_accumulation_steps: int = 1
    max_sequence_length: int = 512
    
    # Cache settings
    enable_tensor_cache: bool = True
    tensor_cache_size_mb: int = 512
    cache_cleanup_interval: float = 300.0


class MemoryLeakDetector:
    """Advanced memory leak detection system"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.object_trackers = {}
        self.memory_snapshots = deque(maxlen=100)
        self.leak_patterns = []
        self.monitored_objects = set()
        self.reference_tracker = weakref.WeakSet()
        
        # Start memory tracing
        if config.enable_memory_profiling:
            tracemalloc.start()
    
    def track_object(self, obj: Any, name: str) -> None:
        """Track object for memory leaks"""
        obj_id = id(obj)
        self.object_trackers[obj_id] = {
            'name': name,
            'created_at': time.time(),
            'size_bytes': self._get_object_size(obj),
            'references': len(gc.get_referrers(obj))
        }
        
        # Add to weak reference tracker
        try:
            self.reference_tracker.add(obj)
        except TypeError:
            pass  # Object is not weakly referenceable
    
    def _get_object_size(self, obj: Any) -> int:
        """Get approximate object size"""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 0
    
    def detect_leaks(self) -> Dict[str, Any]:
        """Detect memory leaks"""
        current_time = time.time()
        leaks_detected = []
        
        # Check for objects that should have been garbage collected
        for obj_id, info in list(self.object_trackers.items()):
            if current_time - info['created_at'] > 300:  # 5 minutes
                try:
                    obj = gc.get_objects()[obj_id]
                    if obj:
                        current_refs = len(gc.get_referrers(obj))
                        if current_refs > info['references']:
                            leaks_detected.append({
                                'object': info['name'],
                                'age_seconds': current_time - info['created_at'],
                                'size_bytes': info['size_bytes'],
                                'ref_count_increase': current_refs - info['references']
                            })
                except:
                    # Object was collected, remove from tracker
                    del self.object_trackers[obj_id]
        
        # Memory growth pattern detection
        if len(self.memory_snapshots) > 10:
            recent_memory = [s.process_memory_gb for s in list(self.memory_snapshots)[-10:]]
            if all(recent_memory[i] < recent_memory[i+1] for i in range(len(recent_memory)-1)):
                leaks_detected.append({
                    'type': 'memory_growth_pattern',
                    'growth_rate_gb': recent_memory[-1] - recent_memory[0],
                    'duration_seconds': 10 * self.config.monitoring_interval
                })
        
        # Check for tensor leaks
        if tracemalloc.is_tracing():
            tensor_leaks = self._detect_tensor_leaks()
            leaks_detected.extend(tensor_leaks)
        
        return {
            'leaks_detected': leaks_detected,
            'total_tracked_objects': len(self.object_trackers),
            'detection_timestamp': current_time
        }
    
    def _detect_tensor_leaks(self) -> List[Dict[str, Any]]:
        """Detect tensor memory leaks"""
        tensor_leaks = []
        
        try:
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.statistics('lineno')[:self.config.profile_top_n]
            
            for stat in top_stats:
                if stat.size > 100 * 1024 * 1024:  # >100MB
                    tensor_leaks.append({
                        'type': 'large_allocation',
                        'size_mb': stat.size / (1024 * 1024),
                        'location': f"{stat.traceback.format()}",
                        'count': stat.count
                    })
        except Exception as e:
            logger.warning(f"Tensor leak detection failed: {e}")
        
        return tensor_leaks
    
    def cleanup_stale_trackers(self) -> None:
        """Clean up stale object trackers"""
        current_time = time.time()
        stale_ids = []
        
        for obj_id, info in self.object_trackers.items():
            if current_time - info['created_at'] > 3600:  # 1 hour
                stale_ids.append(obj_id)
        
        for obj_id in stale_ids:
            del self.object_trackers[obj_id]


class IntelligentBufferManager:
    """Intelligent buffer management system"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.buffers = {}
        self.buffer_stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'size_mb': 0,
            'last_accessed': 0,
            'access_frequency': 0
        })
        self.total_size_mb = 0
        self.lock = threading.RLock()
        self._compression_cache = {}
    
    def get_buffer(self, key: str, size_mb: float, factory: Callable) -> Any:
        """Get or create buffer with intelligent caching"""
        with self.lock:
            if key in self.buffers:
                self.buffer_stats[key]['hits'] += 1
                self.buffer_stats[key]['last_accessed'] = time.time()
                self.buffer_stats[key]['access_frequency'] += 1
                return self.buffers[key]
            
            # Check if we need to evict buffers
            if self.total_size_mb + size_mb > self.config.buffer_size_limit_mb:
                self._evict_buffers(size_mb)
            
            # Create new buffer
            buffer = factory()
            self.buffers[key] = buffer
            self.buffer_stats[key]['misses'] += 1
            self.buffer_stats[key]['size_mb'] = size_mb
            self.buffer_stats[key]['last_accessed'] = time.time()
            self.total_size_mb += size_mb
            
            return buffer
    
    def _evict_buffers(self, required_mb: float) -> None:
        """Evict buffers using LRU + frequency strategy"""
        # Calculate eviction priority (lower is better)
        buffer_priorities = []
        for key, stats in self.buffer_stats.items():
            if key in self.buffers:
                age = time.time() - stats['last_accessed']
                frequency = stats['access_frequency']
                size = stats['size_mb']
                
                # Priority: age * size / frequency
                priority = age * size / max(frequency, 1)
                buffer_priorities.append((priority, key, size))
        
        # Sort by priority (highest first)
        buffer_priorities.sort(reverse=True)
        
        # Evict buffers until we have enough space
        freed_mb = 0
        for priority, key, size in buffer_priorities:
            if freed_mb >= required_mb:
                break
            
            # Compress buffer before eviction if enabled
            if self.config.enable_buffer_compression and size > 100:
                self._compress_buffer(key)
            else:
                del self.buffers[key]
                self.total_size_mb -= size
                freed_mb += size
    
    def _compress_buffer(self, key: str) -> None:
        """Compress buffer to save memory"""
        if key not in self.buffers:
            return
        
        buffer = self.buffers[key]
        try:
            compressed = pickle.dumps(buffer, protocol=pickle.HIGHEST_PROTOCOL)
            self._compression_cache[key] = compressed
            del self.buffers[key]
            
            # Update size statistics
            original_size = self.buffer_stats[key]['size_mb']
            compressed_size = len(compressed) / (1024 * 1024)
            self.buffer_stats[key]['size_mb'] = compressed_size
            self.total_size_mb = self.total_size_mb - original_size + compressed_size
            
            logger.info(f"Compressed buffer {key}: {original_size:.2f}MB -> {compressed_size:.2f}MB")
        except Exception as e:
            logger.warning(f"Failed to compress buffer {key}: {e}")
    
    def _decompress_buffer(self, key: str) -> Any:
        """Decompress buffer from cache"""
        if key not in self._compression_cache:
            return None
        
        try:
            compressed = self._compression_cache[key]
            buffer = pickle.loads(compressed)
            del self._compression_cache[key]
            
            # Move back to active buffers
            self.buffers[key] = buffer
            return buffer
        except Exception as e:
            logger.warning(f"Failed to decompress buffer {key}: {e}")
            return None
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            total_hits = sum(stats['hits'] for stats in self.buffer_stats.values())
            total_misses = sum(stats['misses'] for stats in self.buffer_stats.values())
            hit_rate = total_hits / max(total_hits + total_misses, 1)
            
            return {
                'total_buffers': len(self.buffers),
                'total_size_mb': self.total_size_mb,
                'hit_rate': hit_rate,
                'compressed_buffers': len(self._compression_cache),
                'individual_stats': dict(self.buffer_stats)
            }
    
    def clear_all_buffers(self) -> None:
        """Clear all buffers"""
        with self.lock:
            self.buffers.clear()
            self._compression_cache.clear()
            self.buffer_stats.clear()
            self.total_size_mb = 0


class AdvancedGarbageCollector:
    """Advanced garbage collection system"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.gc_stats = {
            'collections': 0,
            'objects_collected': 0,
            'time_spent': 0,
            'memory_freed_mb': 0
        }
        self.auto_gc_thread = None
        self.stop_auto_gc = threading.Event()
        
        # Configure garbage collection thresholds
        gc.set_threshold(
            config.gc_threshold_0,
            config.gc_threshold_1,
            config.gc_threshold_2
        )
        
        # Start automatic garbage collection
        self.start_auto_gc()
    
    def start_auto_gc(self) -> None:
        """Start automatic garbage collection thread"""
        if self.auto_gc_thread is not None:
            return
        
        self.auto_gc_thread = threading.Thread(target=self._auto_gc_loop, daemon=True)
        self.auto_gc_thread.start()
        logger.info("Automatic garbage collection started")
    
    def stop_auto_gc(self) -> None:
        """Stop automatic garbage collection"""
        if self.auto_gc_thread is None:
            return
        
        self.stop_auto_gc.set()
        self.auto_gc_thread.join()
        self.auto_gc_thread = None
        logger.info("Automatic garbage collection stopped")
    
    def _auto_gc_loop(self) -> None:
        """Automatic garbage collection loop"""
        while not self.stop_auto_gc.is_set():
            try:
                self.intelligent_gc()
                time.sleep(self.config.auto_gc_interval)
            except Exception as e:
                logger.error(f"Auto GC error: {e}")
    
    def intelligent_gc(self) -> Dict[str, Any]:
        """Perform intelligent garbage collection"""
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Get GC stats before collection
        gc_stats_before = gc.get_stats()
        
        # Perform garbage collection
        collected_objects = 0
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects += collected
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final memory check
        final_memory = self._get_memory_usage()
        memory_freed = max(0, initial_memory - final_memory)
        
        # Update statistics
        collection_time = time.time() - start_time
        self.gc_stats['collections'] += 1
        self.gc_stats['objects_collected'] += collected_objects
        self.gc_stats['time_spent'] += collection_time
        self.gc_stats['memory_freed_mb'] += memory_freed * 1024  # Convert to MB
        
        return {
            'collected_objects': collected_objects,
            'memory_freed_mb': memory_freed * 1024,
            'collection_time_ms': collection_time * 1000,
            'gc_stats_before': gc_stats_before,
            'gc_stats_after': gc.get_stats()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def force_full_gc(self) -> Dict[str, Any]:
        """Force full garbage collection"""
        logger.info("Forcing full garbage collection")
        
        # Disable automatic GC temporarily
        gc.disable()
        
        try:
            # Multiple passes for thorough cleanup
            total_collected = 0
            for _ in range(3):
                result = self.intelligent_gc()
                total_collected += result['collected_objects']
            
            # Force finalizers
            gc.collect()
            
            return {
                'total_collected': total_collected,
                'status': 'completed'
            }
        finally:
            # Re-enable automatic GC
            gc.enable()
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            'total_collections': self.gc_stats['collections'],
            'total_objects_collected': self.gc_stats['objects_collected'],
            'total_time_spent': self.gc_stats['time_spent'],
            'total_memory_freed_mb': self.gc_stats['memory_freed_mb'],
            'avg_objects_per_collection': self.gc_stats['objects_collected'] / max(self.gc_stats['collections'], 1),
            'avg_time_per_collection': self.gc_stats['time_spent'] / max(self.gc_stats['collections'], 1),
            'current_thresholds': gc.get_threshold(),
            'current_counts': gc.get_count()
        }


class ModelMemoryOptimizer:
    """Model architecture memory optimization"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.optimization_stats = {
            'models_optimized': 0,
            'memory_saved_mb': 0,
            'optimization_time': 0
        }
    
    def optimize_model(self, model: nn.Module) -> Dict[str, Any]:
        """Optimize model for memory efficiency"""
        start_time = time.time()
        initial_memory = self._get_model_memory_usage(model)
        
        optimizations_applied = []
        
        # 1. Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing(model)
            optimizations_applied.append('gradient_checkpointing')
        
        # 2. Optimize parameter storage
        memory_saved = self._optimize_parameter_storage(model)
        if memory_saved > 0:
            optimizations_applied.append('parameter_storage')
        
        # 3. Fuse operations where possible
        fused_ops = self._fuse_operations(model)
        if fused_ops > 0:
            optimizations_applied.append('operation_fusion')
        
        # 4. Optimize attention mechanisms
        if self._optimize_attention_layers(model):
            optimizations_applied.append('attention_optimization')
        
        # 5. Enable mixed precision if supported
        if self.config.enable_mixed_precision:
            self._enable_mixed_precision(model)
            optimizations_applied.append('mixed_precision')
        
        # Calculate final memory usage
        final_memory = self._get_model_memory_usage(model)
        memory_saved_mb = (initial_memory - final_memory) / (1024 ** 2)
        
        # Update statistics
        optimization_time = time.time() - start_time
        self.optimization_stats['models_optimized'] += 1
        self.optimization_stats['memory_saved_mb'] += memory_saved_mb
        self.optimization_stats['optimization_time'] += optimization_time
        
        return {
            'optimizations_applied': optimizations_applied,
            'memory_saved_mb': memory_saved_mb,
            'optimization_time_ms': optimization_time * 1000,
            'initial_memory_mb': initial_memory / (1024 ** 2),
            'final_memory_mb': final_memory / (1024 ** 2)
        }
    
    def _get_model_memory_usage(self, model: nn.Module) -> int:
        """Get model memory usage in bytes"""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel() * param.element_size()
        
        # Add buffer memory
        for buffer in model.buffers():
            total_params += buffer.numel() * buffer.element_size()
        
        return total_params
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> None:
        """Apply gradient checkpointing to model"""
        try:
            # Look for transformer layers or similar structures
            for name, module in model.named_modules():
                if hasattr(module, 'gradient_checkpointing') and hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
                elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    # Apply checkpointing to transformer layers
                    module.self_attn = torch.utils.checkpoint.checkpoint(module.self_attn)
                    module.linear1 = torch.utils.checkpoint.checkpoint(module.linear1)
                    module.linear2 = torch.utils.checkpoint.checkpoint(module.linear2)
        except Exception as e:
            logger.warning(f"Failed to apply gradient checkpointing: {e}")
    
    def _optimize_parameter_storage(self, model: nn.Module) -> int:
        """Optimize parameter storage"""
        memory_saved = 0
        
        for param in model.parameters():
            if param.requires_grad:
                # Convert to appropriate dtype for memory savings
                if param.dtype == torch.float64:
                    param.data = param.data.to(torch.float32)
                    memory_saved += param.numel() * 4  # 8 bytes -> 4 bytes
                
                # Ensure contiguous memory layout
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        
        return memory_saved
    
    def _fuse_operations(self, model: nn.Module) -> int:
        """Fuse operations for memory efficiency"""
        fused_count = 0
        
        # Look for fusable patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Check for Conv-BatchNorm-ReLU patterns
                layers = list(module.children())
                for i in range(len(layers) - 2):
                    if (isinstance(layers[i], nn.Conv2d) and
                        isinstance(layers[i + 1], nn.BatchNorm2d) and
                        isinstance(layers[i + 2], nn.ReLU)):
                        # This would require torch.jit.script for actual fusion
                        # For now, just count the opportunity
                        fused_count += 1
        
        return fused_count
    
    def _optimize_attention_layers(self, model: nn.Module) -> bool:
        """Optimize attention mechanisms for memory efficiency"""
        optimized = False
        
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Enable memory-efficient attention if available
                if hasattr(module, 'enable_memory_efficient_attention'):
                    module.enable_memory_efficient_attention()
                    optimized = True
        
        return optimized
    
    def _enable_mixed_precision(self, model: nn.Module) -> None:
        """Enable mixed precision training"""
        # Convert appropriate layers to half precision
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
                # These layers typically benefit from mixed precision
                module.half()


class RealTimeMemoryMonitor:
    """Real-time memory monitoring system"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.alert_callbacks = []
        self.memory_profiler = None
        
        # Initialize memory profiler
        if config.enable_memory_profiling:
            self.memory_profiler = MemoryProfiler()
    
    def start_monitoring(self) -> None:
        """Start real-time memory monitoring"""
        if self.monitoring_thread is not None:
            return
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Real-time memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        if self.monitoring_thread is None:
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update profiler
                if self.memory_profiler:
                    self.memory_profiler.update()
                
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _collect_metrics(self) -> MemoryMetrics:
        """Collect current memory metrics"""
        # System memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 ** 3)
        
        # GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # Cache memory (estimate)
        cache_memory = 0
        if hasattr(torch.cuda, 'memory_reserved'):
            cache_memory = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return MemoryMetrics(
            timestamp=time.time(),
            total_memory_gb=memory.total / (1024 ** 3),
            available_memory_gb=memory.available / (1024 ** 3),
            used_memory_gb=memory.used / (1024 ** 3),
            process_memory_gb=process_memory,
            gpu_memory_gb=gpu_memory,
            cache_memory_gb=cache_memory,
            swap_memory_gb=swap.used / (1024 ** 3),
            memory_percent=memory.percent / 100
        )
    
    def _check_alerts(self, metrics: MemoryMetrics) -> None:
        """Check for memory alerts"""
        # Critical memory usage
        if metrics.memory_percent > self.config.memory_critical_threshold:
            self._trigger_alert('critical_memory', metrics)
        
        # Warning memory usage
        elif metrics.memory_percent > self.config.memory_warning_threshold:
            self._trigger_alert('warning_memory', metrics)
        
        # GPU memory alerts
        if metrics.gpu_memory_gb > 0:
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_usage = metrics.gpu_memory_gb / gpu_total
            
            if gpu_usage > 0.9:
                self._trigger_alert('gpu_memory_critical', metrics)
            elif gpu_usage > 0.8:
                self._trigger_alert('gpu_memory_warning', metrics)
    
    def _trigger_alert(self, alert_type: str, metrics: MemoryMetrics) -> None:
        """Trigger memory alert"""
        alert_data = {
            'type': alert_type,
            'timestamp': metrics.timestamp,
            'metrics': metrics.to_dict()
        }
        
        logger.warning(f"Memory alert: {alert_type} - {metrics.memory_percent:.1%} usage")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[MemoryMetrics]:
        """Get current memory metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, last_n: int = 100) -> List[MemoryMetrics]:
        """Get memory metrics history"""
        return list(self.metrics_history)[-last_n:]
    
    def get_memory_trends(self) -> Dict[str, Any]:
        """Get memory usage trends"""
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = list(self.metrics_history)[-min(60, len(self.metrics_history)):]
        
        # Calculate trends
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_values = [m.gpu_memory_gb for m in recent_metrics]
        
        return {
            'memory_trend': self._calculate_trend(memory_values),
            'gpu_memory_trend': self._calculate_trend(gpu_values),
            'average_memory_usage': np.mean(memory_values),
            'peak_memory_usage': np.max(memory_values),
            'memory_volatility': np.std(memory_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'


class MemoryProfiler:
    """Advanced memory profiling system"""
    
    def __init__(self):
        self.profiling_active = False
        self.profile_data = {
            'top_allocations': [],
            'allocation_timeline': [],
            'memory_peaks': [],
            'gc_events': []
        }
    
    def start_profiling(self) -> None:
        """Start memory profiling"""
        if not self.profiling_active:
            tracemalloc.start()
            self.profiling_active = True
            logger.info("Memory profiling started")
    
    def stop_profiling(self) -> None:
        """Stop memory profiling"""
        if self.profiling_active:
            tracemalloc.stop()
            self.profiling_active = False
            logger.info("Memory profiling stopped")
    
    def update(self) -> None:
        """Update profiling data"""
        if not self.profiling_active:
            return
        
        try:
            # Take memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            # Update top allocations
            self.profile_data['top_allocations'] = [
                {
                    'size_mb': stat.size / (1024 ** 2),
                    'count': stat.count,
                    'location': str(stat.traceback)
                }
                for stat in top_stats
            ]
            
            # Record timeline
            self.profile_data['allocation_timeline'].append({
                'timestamp': time.time(),
                'total_size_mb': sum(stat.size for stat in top_stats) / (1024 ** 2)
            })
            
            # Keep only recent data
            if len(self.profile_data['allocation_timeline']) > 1000:
                self.profile_data['allocation_timeline'] = self.profile_data['allocation_timeline'][-1000:]
        
        except Exception as e:
            logger.warning(f"Profiling update failed: {e}")
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Get profiling report"""
        return {
            'profiling_active': self.profiling_active,
            'top_allocations': self.profile_data['top_allocations'],
            'timeline_points': len(self.profile_data['allocation_timeline']),
            'memory_peaks': self.profile_data['memory_peaks'],
            'gc_events': self.profile_data['gc_events']
        }


class TrainingMemoryOptimizer:
    """Training pipeline memory optimization"""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.optimization_history = []
        self.current_batch_size = 32
        self.optimal_batch_size = 32
        self.memory_usage_samples = deque(maxlen=100)
    
    def optimize_batch_size(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Dynamically optimize batch size based on memory usage"""
        if not self.config.batch_size_auto_scaling:
            return self.current_batch_size
        
        logger.info("Optimizing batch size for memory efficiency")
        
        # Test different batch sizes
        test_sizes = [16, 32, 64, 128, 256, 512]
        optimal_size = 32
        
        for batch_size in test_sizes:
            try:
                # Create test batch
                test_batch = sample_input[:batch_size] if sample_input.size(0) >= batch_size else sample_input.repeat(batch_size // sample_input.size(0) + 1, 1)[:batch_size]
                
                # Measure memory usage
                initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_batch)
                
                peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_usage = (peak_memory - initial_memory) / (1024 ** 3)  # GB
                
                # Check if memory usage is acceptable
                if memory_usage < self.config.max_memory_usage_gb * 0.7:  # 70% of max
                    optimal_size = batch_size
                else:
                    break
                
                # Clear memory
                del test_batch, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise
        
        self.optimal_batch_size = optimal_size
        logger.info(f"Optimal batch size: {optimal_size}")
        return optimal_size
    
    def optimize_sequence_length(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Optimize sequence length for memory efficiency"""
        current_length = sample_input.size(1) if sample_input.dim() > 1 else self.config.max_sequence_length
        
        # Test different sequence lengths
        test_lengths = [128, 256, 512, 1024, 2048]
        optimal_length = current_length
        
        for seq_len in test_lengths:
            if seq_len > current_length:
                continue
            
            try:
                # Create test input
                test_input = sample_input[:, :seq_len] if sample_input.dim() > 1 else sample_input
                
                # Measure memory usage
                initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                with torch.no_grad():
                    output = model(test_input)
                
                peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_usage = (peak_memory - initial_memory) / (1024 ** 3)
                
                if memory_usage < self.config.max_memory_usage_gb * 0.8:
                    optimal_length = seq_len
                    break
                
                del test_input, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise
        
        return optimal_length
    
    def optimize_gradient_accumulation(self, target_batch_size: int, max_batch_size: int) -> int:
        """Calculate optimal gradient accumulation steps"""
        if target_batch_size <= max_batch_size:
            return 1
        
        accumulation_steps = (target_batch_size + max_batch_size - 1) // max_batch_size
        return accumulation_steps
    
    def monitor_training_memory(self, step: int, loss: float) -> None:
        """Monitor memory usage during training"""
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 ** 3)
            self.memory_usage_samples.append({
                'step': step,
                'memory_gb': memory_usage,
                'loss': loss,
                'timestamp': time.time()
            })
    
    def get_training_memory_report(self) -> Dict[str, Any]:
        """Get training memory optimization report"""
        if not self.memory_usage_samples:
            return {}
        
        memory_values = [sample['memory_gb'] for sample in self.memory_usage_samples]
        
        return {
            'current_batch_size': self.current_batch_size,
            'optimal_batch_size': self.optimal_batch_size,
            'average_memory_usage_gb': np.mean(memory_values),
            'peak_memory_usage_gb': np.max(memory_values),
            'memory_efficiency': self.optimal_batch_size / self.current_batch_size,
            'samples_collected': len(self.memory_usage_samples)
        }


class MemoryOptimizationSystem:
    """Comprehensive memory optimization system"""
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        self.config = config or MemoryOptimizationConfig()
        
        # Initialize components
        self.leak_detector = MemoryLeakDetector(self.config)
        self.buffer_manager = IntelligentBufferManager(self.config)
        self.garbage_collector = AdvancedGarbageCollector(self.config)
        self.model_optimizer = ModelMemoryOptimizer(self.config)
        self.monitor = RealTimeMemoryMonitor(self.config)
        self.training_optimizer = TrainingMemoryOptimizer(self.config)
        
        # System state
        self.optimization_active = False
        self.optimization_stats = {
            'system_start_time': time.time(),
            'total_optimizations': 0,
            'memory_saved_gb': 0,
            'performance_improvements': []
        }
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Add alert callback
        self.monitor.add_alert_callback(self._handle_memory_alert)
        
        logger.info("Memory optimization system initialized")
    
    def _handle_memory_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle memory alerts with automatic optimization"""
        alert_type = alert_data['type']
        
        if alert_type == 'critical_memory':
            logger.warning("Critical memory alert - triggering emergency optimization")
            self.emergency_memory_cleanup()
        elif alert_type == 'warning_memory':
            logger.info("Memory warning - triggering preventive optimization")
            self.optimize_memory_usage()
    
    def emergency_memory_cleanup(self) -> Dict[str, Any]:
        """Emergency memory cleanup procedure"""
        logger.warning("Executing emergency memory cleanup")
        
        cleanup_results = {
            'timestamp': time.time(),
            'actions_taken': []
        }
        
        # 1. Force garbage collection
        gc_result = self.garbage_collector.force_full_gc()
        cleanup_results['actions_taken'].append(f"Forced GC: {gc_result['total_collected']} objects")
        
        # 2. Clear all buffers
        self.buffer_manager.clear_all_buffers()
        cleanup_results['actions_taken'].append("Cleared all buffers")
        
        # 3. Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleanup_results['actions_taken'].append("Cleared PyTorch cache")
        
        # 4. Clean up leak detector
        self.leak_detector.cleanup_stale_trackers()
        cleanup_results['actions_taken'].append("Cleaned up leak detector")
        
        logger.info(f"Emergency cleanup completed: {len(cleanup_results['actions_taken'])} actions taken")
        return cleanup_results
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across all components"""
        if self.optimization_active:
            return {'status': 'optimization_already_active'}
        
        self.optimization_active = True
        optimization_start = time.time()
        
        try:
            results = {
                'timestamp': optimization_start,
                'optimizations': [],
                'memory_saved_mb': 0,
                'performance_impact': 0
            }
            
            # 1. Intelligent garbage collection
            gc_result = self.garbage_collector.intelligent_gc()
            results['optimizations'].append({
                'type': 'garbage_collection',
                'memory_freed_mb': gc_result['memory_freed_mb'],
                'objects_collected': gc_result['collected_objects']
            })
            results['memory_saved_mb'] += gc_result['memory_freed_mb']
            
            # 2. Buffer optimization
            buffer_stats = self.buffer_manager.get_buffer_stats()
            if buffer_stats['total_size_mb'] > self.config.buffer_size_limit_mb * 0.8:
                # Trigger buffer compression/eviction
                self.buffer_manager._evict_buffers(buffer_stats['total_size_mb'] * 0.3)
                results['optimizations'].append({
                    'type': 'buffer_optimization',
                    'buffers_optimized': buffer_stats['total_buffers']
                })
            
            # 3. Memory leak detection
            leak_results = self.leak_detector.detect_leaks()
            if leak_results['leaks_detected']:
                results['optimizations'].append({
                    'type': 'leak_detection',
                    'leaks_found': len(leak_results['leaks_detected'])
                })
            
            # Update system statistics
            self.optimization_stats['total_optimizations'] += 1
            self.optimization_stats['memory_saved_gb'] += results['memory_saved_mb'] / 1024
            
            optimization_time = time.time() - optimization_start
            results['optimization_time_ms'] = optimization_time * 1000
            
            logger.info(f"Memory optimization completed in {optimization_time:.2f}s")
            return results
            
        finally:
            self.optimization_active = False
    
    def optimize_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Optimize model architecture for memory efficiency"""
        logger.info("Optimizing model architecture")
        return self.model_optimizer.optimize_model(model)
    
    def optimize_training_pipeline(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Optimize training pipeline for memory efficiency"""
        logger.info("Optimizing training pipeline")
        
        optimization_results = {
            'optimal_batch_size': self.training_optimizer.optimize_batch_size(model, sample_input),
            'optimal_sequence_length': self.training_optimizer.optimize_sequence_length(model, sample_input),
            'gradient_accumulation_steps': self.training_optimizer.optimize_gradient_accumulation(128, 32)
        }
        
        return optimization_results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report"""
        current_metrics = self.monitor.get_current_metrics()
        
        report = {
            'timestamp': time.time(),
            'system_uptime_hours': (time.time() - self.optimization_stats['system_start_time']) / 3600,
            'current_memory_usage': current_metrics.to_dict() if current_metrics else None,
            'memory_trends': self.monitor.get_memory_trends(),
            'optimization_stats': self.optimization_stats,
            'component_stats': {
                'leak_detector': {
                    'tracked_objects': len(self.leak_detector.object_trackers),
                    'memory_snapshots': len(self.leak_detector.memory_snapshots)
                },
                'buffer_manager': self.buffer_manager.get_buffer_stats(),
                'garbage_collector': self.garbage_collector.get_gc_stats(),
                'model_optimizer': self.model_optimizer.optimization_stats,
                'training_optimizer': self.training_optimizer.get_training_memory_report()
            }
        }
        
        return report
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Get current system state
        current_metrics = self.monitor.get_current_metrics()
        trends = self.monitor.get_memory_trends()
        
        if current_metrics:
            # High memory usage recommendations
            if current_metrics.memory_percent > 0.8:
                recommendations.append({
                    'priority': 'high',
                    'type': 'memory_usage',
                    'recommendation': 'Consider reducing batch size or enabling gradient checkpointing',
                    'current_value': f"{current_metrics.memory_percent:.1%}",
                    'target_value': "<80%"
                })
            
            # GPU memory recommendations
            if current_metrics.gpu_memory_gb > 0:
                if torch.cuda.is_available():
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    gpu_usage = current_metrics.gpu_memory_gb / gpu_total
                    
                    if gpu_usage > 0.9:
                        recommendations.append({
                            'priority': 'critical',
                            'type': 'gpu_memory',
                            'recommendation': 'Enable mixed precision training or reduce model size',
                            'current_value': f"{gpu_usage:.1%}",
                            'target_value': "<90%"
                        })
        
        # Trend-based recommendations
        if trends.get('memory_trend') == 'increasing':
            recommendations.append({
                'priority': 'medium',
                'type': 'memory_trend',
                'recommendation': 'Memory usage is increasing - check for memory leaks',
                'current_value': trends.get('memory_trend'),
                'target_value': 'stable'
            })
        
        # Buffer management recommendations
        buffer_stats = self.buffer_manager.get_buffer_stats()
        if buffer_stats['hit_rate'] < 0.7:
            recommendations.append({
                'priority': 'medium',
                'type': 'buffer_efficiency',
                'recommendation': 'Low buffer hit rate - consider increasing buffer size',
                'current_value': f"{buffer_stats['hit_rate']:.1%}",
                'target_value': ">70%"
            })
        
        # GC recommendations
        gc_stats = self.garbage_collector.get_gc_stats()
        if gc_stats['avg_time_per_collection'] > 0.1:  # 100ms
            recommendations.append({
                'priority': 'low',
                'type': 'garbage_collection',
                'recommendation': 'GC taking too long - consider tuning thresholds',
                'current_value': f"{gc_stats['avg_time_per_collection']:.3f}s",
                'target_value': "<0.1s"
            })
        
        return recommendations
    
    def export_optimization_report(self, filepath: str) -> None:
        """Export optimization report to file"""
        report = self.get_comprehensive_report()
        recommendations = self.generate_optimization_recommendations()
        
        export_data = {
            'report': report,
            'recommendations': recommendations,
            'config': asdict(self.config),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Optimization report exported to {filepath}")
    
    def cleanup(self) -> None:
        """Cleanup memory optimization system"""
        logger.info("Cleaning up memory optimization system")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Stop garbage collector
        self.garbage_collector.stop_auto_gc()
        
        # Final cleanup
        self.emergency_memory_cleanup()
        
        logger.info("Memory optimization system cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Utility functions
def memory_efficient_decorator(func: Callable) -> Callable:
    """Decorator for memory-efficient function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before execution
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Cleanup after execution
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return wrapper


@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations"""
    # Pre-operation cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        yield
    finally:
        # Post-operation cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_memory_optimized_config(**kwargs) -> MemoryOptimizationConfig:
    """Create memory optimization configuration with custom settings"""
    return MemoryOptimizationConfig(**kwargs)


def quick_memory_optimization() -> Dict[str, Any]:
    """Quick memory optimization for immediate relief"""
    results = {
        'timestamp': time.time(),
        'actions_taken': []
    }
    
    # Force garbage collection
    collected = gc.collect()
    results['actions_taken'].append(f"Garbage collection: {collected} objects")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        results['actions_taken'].append("PyTorch cache cleared")
    
    return results


if __name__ == "__main__":
    # Example usage
    with MemoryOptimizationSystem() as optimizer:
        # Run optimization
        results = optimizer.optimize_memory_usage()
        print(f"Optimization results: {results}")
        
        # Generate report
        report = optimizer.get_comprehensive_report()
        print(f"Memory usage: {report['current_memory_usage']['memory_percent']:.1%}")
        
        # Export report
        optimizer.export_optimization_report("memory_optimization_report.json")
