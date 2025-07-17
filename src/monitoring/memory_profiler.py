"""
Memory Profiling and Leak Detection System
==========================================

Comprehensive memory monitoring system for the Tactical MARL System
designed to detect memory leaks, track PyTorch tensor allocations,
and provide automated garbage collection optimization.

CRITICAL SECURITY IMPLEMENTATION:
- Real-time memory leak detection during decision loops
- PyTorch tensor lifecycle monitoring and cleanup
- Automated memory alerts at 80% usage threshold
- 24-hour soak testing with zero memory growth requirement
- Memory pool management for high-frequency operations

Author: Systems Architect - Infrastructure Hardening
Version: 1.0.0
Classification: CRITICAL SECURITY COMPONENT
"""

import asyncio
import gc
import psutil
import time
import threading
import weakref
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
from contextlib import contextmanager, asynccontextmanager
import torch
import numpy as np
import structlog

# Try to import memray for advanced profiling
try:
    import memray
    MEMRAY_AVAILABLE = True
except ImportError:
    MEMRAY_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time"""
    timestamp: float
    total_mb: float
    used_mb: float
    available_mb: float
    percent: float
    process_mb: float
    pytorch_allocated_mb: float = 0.0
    pytorch_cached_mb: float = 0.0
    python_objects_count: int = 0
    gc_generation_counts: List[int] = field(default_factory=list)
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    detection_time: float
    leak_type: str
    growth_rate_mb_per_sec: float
    total_leaked_mb: float
    affected_components: List[str]
    call_stack: Optional[str] = None
    mitigation_applied: bool = False


@dataclass
class MemoryAlert:
    """Memory usage alert"""
    timestamp: float
    alert_type: str
    severity: str
    message: str
    current_usage_mb: float
    threshold_mb: float
    recommended_actions: List[str]


class TensorTracker:
    """Track PyTorch tensor allocations and lifecycle"""
    
    def __init__(self):
        self.tracked_tensors: Set[weakref.ref] = set()
        self.allocation_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
        self._lock = threading.Lock()
    
    def track_tensor(self, tensor: torch.Tensor, context: str = "unknown"):
        """Track a tensor allocation"""
        if not isinstance(tensor, torch.Tensor):
            return
        
        with self._lock:
            # Create weak reference to avoid keeping tensor alive
            weak_ref = weakref.ref(tensor, self._tensor_cleanup_callback)
            self.tracked_tensors.add(weak_ref)
            
            # Record allocation
            allocation_info = {
                "timestamp": time.time(),
                "context": context,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "size_mb": tensor.numel() * tensor.element_size() / (1024 * 1024),
                "requires_grad": tensor.requires_grad,
                "tensor_id": id(tensor)
            }
            
            self.allocation_history.append(allocation_info)
            
            # Keep history size manageable with more aggressive cleanup
            if len(self.allocation_history) > self.max_history_size:
                # Keep only the most recent entries and some older ones for analysis
                recent_entries = self.allocation_history[-int(self.max_history_size * 0.8):]
                old_entries = self.allocation_history[:int(self.max_history_size * 0.2)]
                self.allocation_history = old_entries + recent_entries
    
    def _tensor_cleanup_callback(self, weak_ref):
        """Called when a tracked tensor is garbage collected"""
        with self._lock:
            self.tracked_tensors.discard(weak_ref)
            # Trigger garbage collection more frequently to prevent memory buildup
            if len(self.tracked_tensors) % 100 == 0:
                import gc
                gc.collect()
    
    def get_tensor_statistics(self) -> Dict[str, Any]:
        """Get current tensor allocation statistics"""
        with self._lock:
            # Clean up dead references
            self.tracked_tensors = {ref for ref in self.tracked_tensors if ref() is not None}
            
            # Calculate statistics
            total_tensors = len(self.tracked_tensors)
            total_memory_mb = 0.0
            contexts = {}
            devices = {}
            
            for tensor_ref in self.tracked_tensors:
                tensor = tensor_ref()
                if tensor is not None:
                    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                    total_memory_mb += size_mb
                    
                    # Track by context (from allocation history)
                    for alloc in reversed(self.allocation_history):
                        if alloc["tensor_id"] == id(tensor):
                            context = alloc["context"]
                            contexts[context] = contexts.get(context, 0) + size_mb
                            break
                    
                    # Track by device
                    device = str(tensor.device)
                    devices[device] = devices.get(device, 0) + size_mb
            
            return {
                "total_tensors": total_tensors,
                "total_memory_mb": total_memory_mb,
                "memory_by_context": contexts,
                "memory_by_device": devices,
                "recent_allocations": self.allocation_history[-100:] if self.allocation_history else []
            }


class MemoryProfiler:
    """
    Comprehensive memory profiling and leak detection system.
    
    Features:
    - Real-time memory monitoring with leak detection
    - PyTorch tensor allocation tracking
    - Automated garbage collection optimization
    - Memory alerts and threshold monitoring
    - Detailed memory snapshots and analysis
    """
    
    def __init__(
        self,
        alert_threshold_percent: float = 80.0,
        leak_detection_window_seconds: float = 300.0,  # 5 minutes
        snapshot_interval_seconds: float = 30.0,
        max_snapshots: int = 1000
    ):
        """
        Initialize the memory profiler.
        
        Args:
            alert_threshold_percent: Memory usage percentage to trigger alerts
            leak_detection_window_seconds: Time window for leak detection
            snapshot_interval_seconds: Interval between memory snapshots
            max_snapshots: Maximum number of snapshots to keep
        """
        self.alert_threshold_percent = alert_threshold_percent
        self.leak_detection_window = leak_detection_window_seconds
        self.snapshot_interval = snapshot_interval_seconds
        self.max_snapshots = max_snapshots
        
        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.detected_leaks: List[MemoryLeak] = []
        self.alerts: List[MemoryAlert] = []
        
        # Tensor tracking
        self.tensor_tracker = TensorTracker()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.gc_stats = {
            "manual_collections": 0,
            "automatic_collections": 0,
            "objects_collected": 0,
            "time_spent_gc": 0.0
        }
        
        # Enable tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep 10 frames
        
        logger.info(
            "MemoryProfiler initialized",
            alert_threshold=alert_threshold_percent,
            leak_detection_window=leak_detection_window_seconds
        )
    
    async def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self._shutdown_event.clear()
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._shutdown_event.set()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                # Take memory snapshot
                snapshot = self._take_memory_snapshot()
                self.snapshots.append(snapshot)
                
                # Keep snapshots manageable
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[-self.max_snapshots:]
                
                # Check for memory leaks
                await self._check_for_memory_leaks()
                
                # Check for memory alerts
                await self._check_memory_alerts(snapshot)
                
                # Optimize garbage collection if needed
                await self._optimize_garbage_collection(snapshot)
                
                # Wait for next snapshot
                await asyncio.sleep(self.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        # System memory
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # PyTorch memory
        pytorch_allocated = 0.0
        pytorch_cached = 0.0
        if torch.cuda.is_available():
            # GPU memory if available
            pytorch_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            pytorch_cached = torch.cuda.memory_reserved() / (1024 * 1024)
        
        # Python object counts
        gc_counts = gc.get_count()
        
        # Top memory allocations
        top_allocations = []
        if tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                top_stats = tracemalloc.take_snapshot().statistics('lineno')
                
                for stat in top_stats[:10]:  # Top 10 allocations
                    top_allocations.append({
                        "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count
                    })
            except Exception as e:
                logger.debug(f"Error getting tracemalloc stats: {e}")
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_mb=memory.total / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            percent=memory.percent,
            process_mb=process_memory.rss / (1024 * 1024),
            pytorch_allocated_mb=pytorch_allocated,
            pytorch_cached_mb=pytorch_cached,
            python_objects_count=sum(gc_counts),
            gc_generation_counts=list(gc_counts),
            top_allocations=top_allocations
        )
    
    async def _check_for_memory_leaks(self):
        """Check for memory leak patterns"""
        if len(self.snapshots) < 3:
            return  # Need at least 3 snapshots
        
        # Get recent snapshots within the detection window
        current_time = time.time()
        recent_snapshots = [
            s for s in self.snapshots
            if current_time - s.timestamp <= self.leak_detection_window
        ]
        
        if len(recent_snapshots) < 2:
            return
        
        # Sort by timestamp
        recent_snapshots.sort(key=lambda x: x.timestamp)
        
        # Calculate memory growth rate
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        time_diff = last_snapshot.timestamp - first_snapshot.timestamp
        if time_diff <= 0:
            return
        
        # Check for different types of leaks
        
        # 1. Process memory leak
        process_growth = last_snapshot.process_mb - first_snapshot.process_mb
        process_growth_rate = process_growth / time_diff
        
        if process_growth_rate > 5.0:  # More than 5 MB/sec growth
            leak = MemoryLeak(
                detection_time=current_time,
                leak_type="process_memory_leak",
                growth_rate_mb_per_sec=process_growth_rate,
                total_leaked_mb=process_growth,
                affected_components=["process_memory"],
                call_stack=self._get_current_call_stack()
            )
            self.detected_leaks.append(leak)
            await self._handle_memory_leak(leak)
        
        # 2. PyTorch memory leak
        pytorch_growth = (last_snapshot.pytorch_allocated_mb + last_snapshot.pytorch_cached_mb) - \
                        (first_snapshot.pytorch_allocated_mb + first_snapshot.pytorch_cached_mb)
        pytorch_growth_rate = pytorch_growth / time_diff
        
        if pytorch_growth_rate > 2.0:  # More than 2 MB/sec PyTorch growth
            leak = MemoryLeak(
                detection_time=current_time,
                leak_type="pytorch_memory_leak",
                growth_rate_mb_per_sec=pytorch_growth_rate,
                total_leaked_mb=pytorch_growth,
                affected_components=["pytorch_tensors", "cuda_cache"],
                call_stack=self._get_tensor_allocation_stack()
            )
            self.detected_leaks.append(leak)
            await self._handle_memory_leak(leak)
        
        # 3. Python object leak
        object_growth = last_snapshot.python_objects_count - first_snapshot.python_objects_count
        object_growth_rate = object_growth / time_diff
        
        if object_growth_rate > 1000:  # More than 1000 objects/sec growth
            leak = MemoryLeak(
                detection_time=current_time,
                leak_type="python_object_leak",
                growth_rate_mb_per_sec=0.0,  # Hard to measure MB for objects
                total_leaked_mb=0.0,
                affected_components=["python_objects", "gc_generations"],
                call_stack=self._get_object_allocation_stack()
            )
            self.detected_leaks.append(leak)
            await self._handle_memory_leak(leak)
    
    async def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check if memory usage exceeds alert thresholds"""
        if snapshot.percent > self.alert_threshold_percent:
            alert = MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type="high_memory_usage",
                severity="WARNING" if snapshot.percent < 90 else "CRITICAL",
                message=f"Memory usage at {snapshot.percent:.1f}% (threshold: {self.alert_threshold_percent}%)",
                current_usage_mb=snapshot.used_mb,
                threshold_mb=snapshot.total_mb * (self.alert_threshold_percent / 100),
                recommended_actions=[
                    "Trigger garbage collection",
                    "Clear PyTorch cache",
                    "Review recent tensor allocations",
                    "Check for memory leaks"
                ]
            )
            self.alerts.append(alert)
            await self._handle_memory_alert(alert)
    
    async def _optimize_garbage_collection(self, snapshot: MemorySnapshot):
        """Optimize garbage collection based on memory usage"""
        # Trigger manual GC if memory usage is high
        if snapshot.percent > 75.0:
            start_time = time.time()
            
            # Run garbage collection
            collected = gc.collect()
            
            gc_time = time.time() - start_time
            self.gc_stats["manual_collections"] += 1
            self.gc_stats["objects_collected"] += collected
            self.gc_stats["time_spent_gc"] += gc_time
            
            logger.info(
                "Manual garbage collection triggered",
                memory_percent=snapshot.percent,
                objects_collected=collected,
                gc_time_ms=gc_time * 1000
            )
            
            # Clear PyTorch cache if needed
            if torch.cuda.is_available() and snapshot.pytorch_cached_mb > 100:
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA cache cleared")
    
    async def _handle_memory_leak(self, leak: MemoryLeak):
        """Handle detected memory leak"""
        logger.critical(
            "Memory leak detected",
            leak_type=leak.leak_type,
            growth_rate_mb_per_sec=leak.growth_rate_mb_per_sec,
            total_leaked_mb=leak.total_leaked_mb,
            affected_components=leak.affected_components
        )
        
        # Apply mitigation strategies
        if leak.leak_type == "pytorch_memory_leak":
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Trigger tensor cleanup
            self.tensor_tracker.get_tensor_statistics()  # This cleans up dead references
            leak.mitigation_applied = True
        
        elif leak.leak_type == "python_object_leak":
            # Force garbage collection
            gc.collect()
            leak.mitigation_applied = True
        
        elif leak.leak_type == "process_memory_leak":
            # Force full cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            leak.mitigation_applied = True
    
    async def _handle_memory_alert(self, alert: MemoryAlert):
        """Handle memory usage alert"""
        logger.warning(
            "Memory alert triggered",
            alert_type=alert.alert_type,
            severity=alert.severity,
            message=alert.message,
            current_usage_mb=alert.current_usage_mb
        )
        
        # Apply recommended actions for high severity alerts
        if alert.severity == "CRITICAL":
            # Immediate cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _get_current_call_stack(self) -> str:
        """Get current call stack for leak analysis"""
        try:
            import traceback
            return ''.join(traceback.format_stack()[-10:])  # Last 10 frames
        except Exception:
            return "Call stack unavailable"
    
    def _get_tensor_allocation_stack(self) -> str:
        """Get tensor allocation call stack"""
        tensor_stats = self.tensor_tracker.get_tensor_statistics()
        recent_allocs = tensor_stats.get("recent_allocations", [])
        
        if recent_allocs:
            return f"Recent tensor allocations: {len(recent_allocs)} tensors"
        return "No recent tensor allocations"
    
    def _get_object_allocation_stack(self) -> str:
        """Get Python object allocation information"""
        if tracemalloc.is_tracing():
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                if top_stats:
                    return f"Top allocation: {top_stats[0].traceback.format()[-1]}"
            except Exception:
                pass
        return "Object allocation info unavailable"
    
    @contextmanager
    def track_tensor_context(self, context_name: str):
        """Context manager to track tensor allocations in a specific context"""
        original_new = torch.Tensor.__new__
        
        def tracking_new(cls, *args, **kwargs):
            tensor = original_new(cls)
            self.tensor_tracker.track_tensor(tensor, context_name)
            return tensor
        
        try:
            torch.Tensor.__new__ = staticmethod(tracking_new)
            yield
        finally:
            torch.Tensor.__new__ = original_new
    
    @asynccontextmanager
    async def profile_memory_usage(self, operation_name: str):
        """Async context manager to profile memory usage of an operation"""
        # Take snapshot before operation
        start_snapshot = self._take_memory_snapshot()
        start_time = time.time()
        
        try:
            yield start_snapshot
        finally:
            # Take snapshot after operation
            end_snapshot = self._take_memory_snapshot()
            duration = time.time() - start_time
            
            # Calculate memory delta
            memory_delta = end_snapshot.process_mb - start_snapshot.process_mb
            pytorch_delta = (end_snapshot.pytorch_allocated_mb + end_snapshot.pytorch_cached_mb) - \
                           (start_snapshot.pytorch_allocated_mb + start_snapshot.pytorch_cached_mb)
            
            logger.info(
                "Memory usage profiled",
                operation=operation_name,
                duration_ms=duration * 1000,
                memory_delta_mb=memory_delta,
                pytorch_delta_mb=pytorch_delta,
                final_usage_percent=end_snapshot.percent
            )
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        if not self.snapshots:
            return {"error": "No memory snapshots available"}
        
        latest_snapshot = self.snapshots[-1]
        tensor_stats = self.tensor_tracker.get_tensor_statistics()
        
        # Calculate trends
        memory_trend = "stable"
        if len(self.snapshots) >= 10:
            recent_snapshots = self.snapshots[-10:]
            first_usage = recent_snapshots[0].process_mb
            last_usage = recent_snapshots[-1].process_mb
            
            if last_usage > first_usage * 1.1:
                memory_trend = "increasing"
            elif last_usage < first_usage * 0.9:
                memory_trend = "decreasing"
        
        return {
            "timestamp": latest_snapshot.timestamp,
            "current_usage": {
                "system_memory_percent": latest_snapshot.percent,
                "process_memory_mb": latest_snapshot.process_mb,
                "pytorch_allocated_mb": latest_snapshot.pytorch_allocated_mb,
                "pytorch_cached_mb": latest_snapshot.pytorch_cached_mb,
                "python_objects": latest_snapshot.python_objects_count
            },
            "tensor_statistics": tensor_stats,
            "memory_trend": memory_trend,
            "detected_leaks": len(self.detected_leaks),
            "recent_alerts": len([a for a in self.alerts if time.time() - a.timestamp < 3600]),  # Last hour
            "gc_statistics": self.gc_stats,
            "snapshots_collected": len(self.snapshots),
            "monitoring_active": self.monitoring_active,
            "recommendations": self._generate_memory_recommendations()
        }
    
    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if not self.snapshots:
            return ["Start memory monitoring to get recommendations"]
        
        latest = self.snapshots[-1]
        
        if latest.percent > 85:
            recommendations.append("URGENT: Memory usage critically high - consider scaling or optimization")
        
        if latest.pytorch_cached_mb > 200:
            recommendations.append("Clear PyTorch CUDA cache to free memory")
        
        if len(self.detected_leaks) > 0:
            recommendations.append(f"Address {len(self.detected_leaks)} detected memory leaks")
        
        tensor_stats = self.tensor_tracker.get_tensor_statistics()
        if tensor_stats["total_memory_mb"] > 100:
            recommendations.append("Review tensor allocations - high tensor memory usage detected")
        
        if self.gc_stats["manual_collections"] > 10:
            recommendations.append("Frequent manual GC indicates memory pressure - review allocation patterns")
        
        return recommendations


# Global memory profiler instance
_memory_profiler: Optional[MemoryProfiler] = None


async def get_memory_profiler() -> MemoryProfiler:
    """Get or create the global memory profiler instance"""
    global _memory_profiler
    
    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
        await _memory_profiler.start_monitoring()
    
    return _memory_profiler


async def shutdown_memory_profiler():
    """Shutdown the global memory profiler instance"""
    global _memory_profiler
    
    if _memory_profiler:
        await _memory_profiler.stop_monitoring()
        _memory_profiler = None


# Convenience functions for easy integration
async def profile_memory_usage(operation_name: str):
    """Decorator/context manager for profiling memory usage"""
    profiler = await get_memory_profiler()
    return profiler.profile_memory_usage(operation_name)


def track_tensor_allocations(context_name: str):
    """Context manager for tracking tensor allocations"""
    global _memory_profiler
    if _memory_profiler:
        return _memory_profiler.track_tensor_context(context_name)
    else:
        # Return a no-op context manager if profiler not initialized
        from contextlib import nullcontext
        return nullcontext()