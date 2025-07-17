"""
Advanced Resource Management System for Parallel Test Execution
Agent 2 Mission: CPU Affinity, Memory Limits, and GPU Scheduling

This module provides comprehensive resource management including CPU affinity
optimization, memory usage monitoring, GPU resource scheduling, and
resource contention detection for optimal test execution performance.
"""

import os
import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import subprocess
import queue
import signal
import multiprocessing
from contextlib import contextmanager
import resource

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for test execution"""
    memory_mb: int = 1024
    cpu_percent: float = 100.0
    cpu_cores: Optional[List[int]] = None
    wall_time_seconds: int = 300
    file_descriptors: int = 1024
    gpu_memory_mb: Optional[int] = None
    gpu_id: Optional[int] = None


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    memory_mb: float
    cpu_percent: float
    cpu_cores_used: List[int]
    wall_time_seconds: float
    file_descriptors: int
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerResourceAllocation:
    """Resource allocation for a specific worker"""
    worker_id: str
    process_id: int
    cpu_affinity: List[int]
    memory_limit_mb: int
    gpu_id: Optional[int] = None
    status: str = "idle"  # idle, running, overloaded, terminated
    allocated_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


class SystemResourceMonitor:
    """Monitor system-wide resource usage"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history: List[ResourceUsage] = []
        self.history_lock = threading.Lock()
        self.max_history_size = 1000
        
    def start_monitoring(self):
        """Start system resource monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started system resource monitoring")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped system resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                usage = self._collect_system_usage()
                
                with self.history_lock:
                    self.resource_history.append(usage)
                    if len(self.resource_history) > self.max_history_size:
                        self.resource_history.pop(0)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_usage(self) -> ResourceUsage:
        """Collect current system resource usage"""
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024 * 1024)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cores = [i for i, usage in enumerate(cpu_per_core) if usage > 10.0]
        
        # File descriptors
        try:
            fd_count = len(os.listdir('/proc/self/fd'))
        except (FileNotFoundError, IOError, OSError) as e:
            fd_count = 0
        
        # GPU usage (if available)
        gpu_memory_mb = None
        gpu_utilization = None
        try:
            gpu_info = self._get_gpu_info()
            if gpu_info:
                gpu_memory_mb = gpu_info.get('memory_used_mb')
                gpu_utilization = gpu_info.get('utilization_percent')
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return ResourceUsage(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            cpu_cores_used=active_cores,
            wall_time_seconds=time.time(),
            file_descriptors=fd_count,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization
        )
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    return {
                        'memory_used_mb': float(parts[0]),
                        'memory_total_mb': float(parts[1]),
                        'utilization_percent': float(parts[2])
                    }
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return None
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get current resource usage"""
        with self.history_lock:
            return self.resource_history[-1] if self.resource_history else None
    
    def get_usage_history(self, duration_seconds: int = 300) -> List[ResourceUsage]:
        """Get resource usage history for specified duration"""
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        
        with self.history_lock:
            return [
                usage for usage in self.resource_history
                if usage.timestamp >= cutoff_time
            ]
    
    def detect_resource_contention(self) -> Dict[str, Any]:
        """Detect resource contention patterns"""
        recent_usage = self.get_usage_history(60)  # Last minute
        
        if not recent_usage:
            return {"contention_detected": False}
        
        # Analyze patterns
        cpu_high_count = sum(1 for u in recent_usage if u.cpu_percent > 80)
        memory_high_count = sum(1 for u in recent_usage if u.memory_mb > psutil.virtual_memory().total * 0.8 / (1024 * 1024))
        
        cpu_contention = cpu_high_count > len(recent_usage) * 0.7
        memory_contention = memory_high_count > len(recent_usage) * 0.7
        
        return {
            "contention_detected": cpu_contention or memory_contention,
            "cpu_contention": cpu_contention,
            "memory_contention": memory_contention,
            "avg_cpu_usage": sum(u.cpu_percent for u in recent_usage) / len(recent_usage),
            "avg_memory_usage": sum(u.memory_mb for u in recent_usage) / len(recent_usage),
            "recommendation": self._get_contention_recommendation(cpu_contention, memory_contention)
        }
    
    def _get_contention_recommendation(self, cpu_contention: bool, memory_contention: bool) -> str:
        """Get recommendation for resource contention"""
        if cpu_contention and memory_contention:
            return "Reduce parallel workers and implement memory limits"
        elif cpu_contention:
            return "Reduce parallel workers or implement CPU affinity"
        elif memory_contention:
            return "Implement memory limits or reduce memory-intensive tests"
        else:
            return "No action needed"


class CPUAffinityManager:
    """Manage CPU affinity for optimal performance"""
    
    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.worker_assignments: Dict[str, List[int]] = {}
        self.cpu_usage_tracking: Dict[int, List[str]] = defaultdict(list)
        self.assignment_lock = threading.Lock()
        
    def assign_cpu_cores(self, worker_id: str, preferred_cores: Optional[List[int]] = None) -> List[int]:
        """Assign CPU cores to a worker"""
        with self.assignment_lock:
            if worker_id in self.worker_assignments:
                return self.worker_assignments[worker_id]
            
            if preferred_cores:
                assigned_cores = preferred_cores
            else:
                assigned_cores = self._find_optimal_cores(worker_id)
            
            self.worker_assignments[worker_id] = assigned_cores
            
            for core in assigned_cores:
                self.cpu_usage_tracking[core].append(worker_id)
            
            return assigned_cores
    
    def _find_optimal_cores(self, worker_id: str) -> List[int]:
        """Find optimal CPU cores for a worker"""
        # Strategy: Assign cores with least current usage
        core_usage = {core: len(workers) for core, workers in self.cpu_usage_tracking.items()}
        
        # Add unused cores
        for core in range(self.cpu_count):
            if core not in core_usage:
                core_usage[core] = 0
        
        # Sort by usage (ascending)
        sorted_cores = sorted(core_usage.items(), key=lambda x: x[1])
        
        # Assign 1-2 cores based on system capacity
        cores_per_worker = max(1, self.cpu_count // 4)
        assigned_cores = [core for core, _ in sorted_cores[:cores_per_worker]]
        
        return assigned_cores
    
    def set_cpu_affinity(self, worker_id: str, process_id: Optional[int] = None) -> bool:
        """Set CPU affinity for a worker process"""
        if worker_id not in self.worker_assignments:
            return False
        
        try:
            assigned_cores = self.worker_assignments[worker_id]
            
            if process_id:
                process = psutil.Process(process_id)
            else:
                process = psutil.Process()
            
            process.cpu_affinity(assigned_cores)
            logger.info(f"Set CPU affinity for worker {worker_id} to cores {assigned_cores}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set CPU affinity for worker {worker_id}: {e}")
            return False
    
    def release_worker_cores(self, worker_id: str):
        """Release CPU cores assigned to a worker"""
        with self.assignment_lock:
            if worker_id in self.worker_assignments:
                assigned_cores = self.worker_assignments[worker_id]
                
                for core in assigned_cores:
                    if worker_id in self.cpu_usage_tracking[core]:
                        self.cpu_usage_tracking[core].remove(worker_id)
                
                del self.worker_assignments[worker_id]
                logger.info(f"Released CPU cores for worker {worker_id}")
    
    def get_cpu_assignments(self) -> Dict[str, List[int]]:
        """Get current CPU core assignments"""
        with self.assignment_lock:
            return self.worker_assignments.copy()


class MemoryLimitManager:
    """Manage memory limits for test processes"""
    
    def __init__(self):
        self.memory_limits: Dict[str, int] = {}
        self.memory_usage: Dict[str, List[float]] = defaultdict(list)
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        self.limits_lock = threading.Lock()
    
    def set_memory_limit(self, worker_id: str, limit_mb: int) -> bool:
        """Set memory limit for a worker"""
        with self.limits_lock:
            self.memory_limits[worker_id] = limit_mb
            
            try:
                # Set soft and hard limits
                soft_limit = limit_mb * 1024 * 1024  # Convert to bytes
                hard_limit = soft_limit * 1.2  # 20% buffer
                
                resource.setrlimit(resource.RLIMIT_AS, (int(soft_limit), int(hard_limit)))
                logger.info(f"Set memory limit for worker {worker_id}: {limit_mb}MB")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set memory limit for worker {worker_id}: {e}")
                return False
    
    def check_memory_usage(self, worker_id: str, process_id: Optional[int] = None) -> Dict[str, Any]:
        """Check current memory usage for a worker"""
        try:
            if process_id:
                process = psutil.Process(process_id)
            else:
                process = psutil.Process()
            
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Track usage history
            with self.limits_lock:
                self.memory_usage[worker_id].append(memory_mb)
                if len(self.memory_usage[worker_id]) > 100:  # Keep last 100 measurements
                    self.memory_usage[worker_id].pop(0)
                
                limit_mb = self.memory_limits.get(worker_id, float('inf'))
                
                return {
                    "worker_id": worker_id,
                    "current_memory_mb": memory_mb,
                    "limit_mb": limit_mb,
                    "usage_percent": (memory_mb / limit_mb) * 100 if limit_mb != float('inf') else 0,
                    "within_limit": memory_mb <= limit_mb,
                    "peak_memory_mb": max(self.memory_usage[worker_id]) if self.memory_usage[worker_id] else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to check memory usage for worker {worker_id}: {e}")
            return {
                "worker_id": worker_id,
                "error": str(e),
                "within_limit": False
            }
    
    def get_memory_recommendations(self, worker_id: str) -> Dict[str, Any]:
        """Get memory optimization recommendations"""
        with self.limits_lock:
            if worker_id not in self.memory_usage or not self.memory_usage[worker_id]:
                return {"error": "No memory usage data available"}
            
            usage_history = self.memory_usage[worker_id]
            avg_usage = sum(usage_history) / len(usage_history)
            peak_usage = max(usage_history)
            limit_mb = self.memory_limits.get(worker_id, float('inf'))
            
            recommendations = []
            
            # Peak usage analysis
            if peak_usage > limit_mb * 0.9:
                recommendations.append({
                    "type": "limit_increase",
                    "severity": "high",
                    "message": f"Peak usage ({peak_usage:.1f}MB) near limit ({limit_mb}MB)",
                    "suggestion": f"Consider increasing limit to {peak_usage * 1.2:.0f}MB"
                })
            
            # Average usage analysis
            if avg_usage > limit_mb * 0.7:
                recommendations.append({
                    "type": "optimization",
                    "severity": "medium",
                    "message": f"High average usage ({avg_usage:.1f}MB)",
                    "suggestion": "Consider memory optimization or test isolation"
                })
            
            # Trend analysis
            if len(usage_history) > 10:
                recent_avg = sum(usage_history[-10:]) / 10
                if recent_avg > avg_usage * 1.2:
                    recommendations.append({
                        "type": "memory_leak",
                        "severity": "high",
                        "message": "Memory usage trending upward",
                        "suggestion": "Check for memory leaks in tests"
                    })
            
            return {
                "worker_id": worker_id,
                "avg_usage_mb": avg_usage,
                "peak_usage_mb": peak_usage,
                "limit_mb": limit_mb,
                "recommendations": recommendations
            }


class GPUResourceManager:
    """Manage GPU resources for AI/ML tests"""
    
    def __init__(self):
        self.gpu_assignments: Dict[str, int] = {}
        self.gpu_memory_limits: Dict[str, int] = {}
        self.available_gpus = self._detect_gpus()
        self.assignment_lock = threading.Lock()
    
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': float(parts[2]),
                            'memory_free_mb': float(parts[3])
                        })
                return gpus
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return []
    
    def assign_gpu(self, worker_id: str, memory_mb: Optional[int] = None) -> Optional[int]:
        """Assign GPU to a worker"""
        if not self.available_gpus:
            return None
        
        with self.assignment_lock:
            if worker_id in self.gpu_assignments:
                return self.gpu_assignments[worker_id]
            
            # Find GPU with most free memory
            available_gpus = [
                gpu for gpu in self.available_gpus
                if gpu['index'] not in self.gpu_assignments.values()
            ]
            
            if not available_gpus:
                logger.warning(f"No available GPUs for worker {worker_id}")
                return None
            
            # Select GPU with most free memory
            best_gpu = max(available_gpus, key=lambda x: x['memory_free_mb'])
            gpu_id = best_gpu['index']
            
            self.gpu_assignments[worker_id] = gpu_id
            if memory_mb:
                self.gpu_memory_limits[worker_id] = memory_mb
            
            logger.info(f"Assigned GPU {gpu_id} to worker {worker_id}")
            return gpu_id
    
    def release_gpu(self, worker_id: str):
        """Release GPU assigned to a worker"""
        with self.assignment_lock:
            if worker_id in self.gpu_assignments:
                gpu_id = self.gpu_assignments[worker_id]
                del self.gpu_assignments[worker_id]
                
                if worker_id in self.gpu_memory_limits:
                    del self.gpu_memory_limits[worker_id]
                
                logger.info(f"Released GPU {gpu_id} from worker {worker_id}")
    
    def get_gpu_usage(self, gpu_id: int) -> Optional[Dict[str, Any]]:
        """Get GPU usage statistics"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits', f'--id={gpu_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    'gpu_id': gpu_id,
                    'memory_used_mb': float(parts[0]),
                    'memory_total_mb': float(parts[1]),
                    'utilization_percent': float(parts[2])
                }
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return None
    
    def set_gpu_environment(self, worker_id: str) -> bool:
        """Set GPU environment variables for a worker"""
        with self.assignment_lock:
            if worker_id not in self.gpu_assignments:
                return False
            
            gpu_id = self.gpu_assignments[worker_id]
            
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                # Set memory limit if specified
                if worker_id in self.gpu_memory_limits:
                    memory_limit = self.gpu_memory_limits[worker_id]
                    os.environ['CUDA_MEMORY_LIMIT'] = str(memory_limit)
                
                logger.info(f"Set GPU environment for worker {worker_id}: GPU {gpu_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set GPU environment for worker {worker_id}: {e}")
                return False


class AdvancedResourceManager:
    """Comprehensive resource management system"""
    
    def __init__(self):
        self.system_monitor = SystemResourceMonitor()
        self.cpu_manager = CPUAffinityManager()
        self.memory_manager = MemoryLimitManager()
        self.gpu_manager = GPUResourceManager()
        self.worker_allocations: Dict[str, WorkerResourceAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Start monitoring
        self.system_monitor.start_monitoring()
    
    def allocate_resources(self, worker_id: str, limits: ResourceLimits) -> WorkerResourceAllocation:
        """Allocate resources for a worker"""
        with self.allocation_lock:
            # Assign CPU cores
            cpu_cores = self.cpu_manager.assign_cpu_cores(worker_id, limits.cpu_cores)
            
            # Set memory limit
            self.memory_manager.set_memory_limit(worker_id, limits.memory_mb)
            
            # Assign GPU if requested
            gpu_id = None
            if limits.gpu_memory_mb:
                gpu_id = self.gpu_manager.assign_gpu(worker_id, limits.gpu_memory_mb)
            
            allocation = WorkerResourceAllocation(
                worker_id=worker_id,
                process_id=os.getpid(),
                cpu_affinity=cpu_cores,
                memory_limit_mb=limits.memory_mb,
                gpu_id=gpu_id
            )
            
            self.worker_allocations[worker_id] = allocation
            
            # Apply resource limits
            self._apply_resource_limits(worker_id, limits)
            
            logger.info(f"Allocated resources for worker {worker_id}: CPU={cpu_cores}, Memory={limits.memory_mb}MB, GPU={gpu_id}")
            return allocation
    
    def _apply_resource_limits(self, worker_id: str, limits: ResourceLimits):
        """Apply resource limits to current process"""
        # Set CPU affinity
        self.cpu_manager.set_cpu_affinity(worker_id)
        
        # Set GPU environment
        if limits.gpu_memory_mb:
            self.gpu_manager.set_gpu_environment(worker_id)
        
        # Set process limits
        try:
            # Wall time limit
            if limits.wall_time_seconds:
                resource.setrlimit(resource.RLIMIT_CPU, (limits.wall_time_seconds, limits.wall_time_seconds))
            
            # File descriptor limit
            if limits.file_descriptors:
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits.file_descriptors, limits.file_descriptors))
                
        except Exception as e:
            logger.warning(f"Failed to set some resource limits: {e}")
    
    def monitor_worker_resources(self, worker_id: str) -> Dict[str, Any]:
        """Monitor resource usage for a worker"""
        if worker_id not in self.worker_allocations:
            return {"error": "Worker not found"}
        
        allocation = self.worker_allocations[worker_id]
        
        # Check memory usage
        memory_info = self.memory_manager.check_memory_usage(worker_id, allocation.process_id)
        
        # Check GPU usage
        gpu_info = None
        if allocation.gpu_id is not None:
            gpu_info = self.gpu_manager.get_gpu_usage(allocation.gpu_id)
        
        # Get system contention info
        contention = self.system_monitor.detect_resource_contention()
        
        return {
            "worker_id": worker_id,
            "allocation": allocation,
            "memory_usage": memory_info,
            "gpu_usage": gpu_info,
            "system_contention": contention,
            "timestamp": datetime.now().isoformat()
        }
    
    def release_worker_resources(self, worker_id: str):
        """Release all resources for a worker"""
        with self.allocation_lock:
            if worker_id in self.worker_allocations:
                # Release CPU cores
                self.cpu_manager.release_worker_cores(worker_id)
                
                # Release GPU
                self.gpu_manager.release_gpu(worker_id)
                
                # Remove allocation
                del self.worker_allocations[worker_id]
                
                logger.info(f"Released all resources for worker {worker_id}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get system-wide optimization recommendations"""
        recommendations = []
        
        # System-level recommendations
        contention = self.system_monitor.detect_resource_contention()
        if contention["contention_detected"]:
            recommendations.append({
                "type": "system",
                "severity": "high",
                "message": "Resource contention detected",
                "details": contention,
                "action": contention["recommendation"]
            })
        
        # Worker-specific recommendations
        for worker_id in self.worker_allocations:
            memory_rec = self.memory_manager.get_memory_recommendations(worker_id)
            if memory_rec.get("recommendations"):
                recommendations.extend(memory_rec["recommendations"])
        
        # CPU utilization recommendations
        cpu_assignments = self.cpu_manager.get_cpu_assignments()
        if len(cpu_assignments) > os.cpu_count():
            recommendations.append({
                "type": "cpu",
                "severity": "medium",
                "message": "CPU over-subscription detected",
                "action": "Reduce parallel workers or implement better CPU affinity"
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "system_status": {
                "total_workers": len(self.worker_allocations),
                "cpu_assignments": cpu_assignments,
                "available_gpus": len(self.gpu_manager.available_gpus),
                "system_contention": contention
            }
        }
    
    def shutdown(self):
        """Shutdown the resource manager"""
        # Release all worker resources
        for worker_id in list(self.worker_allocations.keys()):
            self.release_worker_resources(worker_id)
        
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        
        logger.info("Resource manager shutdown complete")


@contextmanager
def managed_resources(worker_id: str, limits: ResourceLimits):
    """Context manager for automatic resource management"""
    manager = AdvancedResourceManager()
    
    try:
        allocation = manager.allocate_resources(worker_id, limits)
        yield allocation
    finally:
        manager.release_worker_resources(worker_id)
        manager.shutdown()


if __name__ == "__main__":
    # Demo usage
    import time
    
    # Create resource limits
    limits = ResourceLimits(
        memory_mb=512,
        cpu_cores=[0, 1],
        wall_time_seconds=60
    )
    
    # Use managed resources
    with managed_resources("test_worker", limits) as allocation:
        print(f"Allocated resources: {allocation}")
        
        # Simulate some work
        time.sleep(2)
        
        # Monitor resources
        manager = AdvancedResourceManager()
        monitoring_info = manager.monitor_worker_resources("test_worker")
        print(f"Resource monitoring: {monitoring_info}")
        
        # Get recommendations
        recommendations = manager.get_optimization_recommendations()
        print(f"Recommendations: {recommendations}")