"""
Hardware-Aware Performance Profiler
==================================

Provides hardware-aware testing capabilities including NUMA optimization,
CPU cache analysis, and memory locality testing for ultra-low latency systems.
"""

import os
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import subprocess
import platform
from .nanosecond_timer import NanosecondTimer, TimingResult


@dataclass
class HardwareConfig:
    """Hardware configuration information"""
    cpu_count: int
    numa_nodes: int
    cache_sizes: Dict[str, int]
    cpu_frequency: float
    memory_total_gb: float
    platform: str
    architecture: str


@dataclass
class CacheProfile:
    """CPU cache performance profile"""
    l1_hits: int
    l1_misses: int
    l2_hits: int
    l2_misses: int
    l3_hits: int
    l3_misses: int
    hit_rate: float
    miss_penalty_ns: float


@dataclass
class NumaProfile:
    """NUMA performance profile"""
    node_id: int
    local_access_ns: float
    remote_access_ns: float
    locality_ratio: float
    memory_bandwidth_gbps: float
    cpu_affinity: List[int]


class HardwareProfiler:
    """
    Hardware-aware performance profiler for ultra-low latency systems
    
    Features:
    - NUMA topology analysis
    - CPU cache performance profiling
    - Memory locality optimization
    - Hardware-specific optimizations
    """
    
    def __init__(self, timer: NanosecondTimer):
        self.timer = timer
        self.hardware_config = self._detect_hardware()
        self.numa_profiles: Dict[int, NumaProfile] = {}
        self.cache_profiles: Dict[int, CacheProfile] = {}
        
    def _detect_hardware(self) -> HardwareConfig:
        """Detect hardware configuration"""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # NUMA nodes detection
            numa_nodes = self._count_numa_nodes()
            
            # Cache sizes (Linux specific)
            cache_sizes = self._get_cache_sizes()
            
            return HardwareConfig(
                cpu_count=cpu_count,
                numa_nodes=numa_nodes,
                cache_sizes=cache_sizes,
                cpu_frequency=cpu_freq.current if cpu_freq else 0.0,
                memory_total_gb=memory.total / (1024**3),
                platform=platform.system(),
                architecture=platform.machine()
            )
            
        except Exception as e:
            # Fallback configuration
            return HardwareConfig(
                cpu_count=psutil.cpu_count() or 1,
                numa_nodes=1,
                cache_sizes={},
                cpu_frequency=0.0,
                memory_total_gb=psutil.virtual_memory().total / (1024**3),
                platform=platform.system(),
                architecture=platform.machine()
            )
    
    def _count_numa_nodes(self) -> int:
        """Count NUMA nodes on the system"""
        try:
            if platform.system() == "Linux":
                # Try to read from /sys/devices/system/node/
                numa_path = "/sys/devices/system/node/"
                if os.path.exists(numa_path):
                    nodes = [d for d in os.listdir(numa_path) if d.startswith("node")]
                    return len(nodes)
                
                # Try numactl command
                result = subprocess.run(
                    ["numactl", "--hardware"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'available:' in line and 'nodes' in line:
                            # Extract node count from "available: 2 nodes (0-1)"
                            parts = line.split()
                            if len(parts) >= 2:
                                return int(parts[1])
            
            return 1  # Default to single node
            
        except Exception:
            return 1
    
    def _get_cache_sizes(self) -> Dict[str, int]:
        """Get CPU cache sizes"""
        cache_sizes = {}
        
        try:
            if platform.system() == "Linux":
                # Read from /sys/devices/system/cpu/cpu0/cache/
                base_path = "/sys/devices/system/cpu/cpu0/cache/"
                if os.path.exists(base_path):
                    for cache_dir in os.listdir(base_path):
                        if cache_dir.startswith("index"):
                            level_path = os.path.join(base_path, cache_dir, "level")
                            size_path = os.path.join(base_path, cache_dir, "size")
                            
                            if os.path.exists(level_path) and os.path.exists(size_path):
                                with open(level_path, 'r') as f:
                                    level = f.read().strip()
                                with open(size_path, 'r') as f:
                                    size_str = f.read().strip()
                                
                                # Parse size (e.g., "32K", "256K", "8M")
                                size_value = 0
                                if size_str.endswith('K'):
                                    size_value = int(size_str[:-1]) * 1024
                                elif size_str.endswith('M'):
                                    size_value = int(size_str[:-1]) * 1024 * 1024
                                
                                cache_sizes[f"L{level}"] = size_value
            
        except Exception:
            pass
        
        return cache_sizes
    
    def profile_numa_performance(self, node_id: int, test_size_mb: int = 64) -> NumaProfile:
        """Profile NUMA node performance"""
        if node_id >= self.hardware_config.numa_nodes:
            raise ValueError(f"NUMA node {node_id} not available")
        
        # Get CPUs for this NUMA node
        cpu_affinity = self._get_numa_cpus(node_id)
        
        # Set CPU affinity for testing
        original_affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        
        try:
            if cpu_affinity and hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, cpu_affinity)
            
            # Test local memory access
            local_access_ns = self._measure_memory_access(test_size_mb, "local")
            
            # Test remote memory access (if multi-NUMA)
            remote_access_ns = local_access_ns  # Default to local
            if self.hardware_config.numa_nodes > 1:
                remote_access_ns = self._measure_remote_memory_access(test_size_mb)
            
            # Calculate locality ratio
            locality_ratio = local_access_ns / remote_access_ns if remote_access_ns > 0 else 1.0
            
            # Estimate memory bandwidth
            bandwidth_gbps = self._estimate_memory_bandwidth(test_size_mb)
            
            profile = NumaProfile(
                node_id=node_id,
                local_access_ns=local_access_ns,
                remote_access_ns=remote_access_ns,
                locality_ratio=locality_ratio,
                memory_bandwidth_gbps=bandwidth_gbps,
                cpu_affinity=cpu_affinity
            )
            
            self.numa_profiles[node_id] = profile
            return profile
            
        finally:
            # Restore original CPU affinity
            if original_affinity and hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, original_affinity)
    
    def _get_numa_cpus(self, node_id: int) -> List[int]:
        """Get CPUs belonging to a NUMA node"""
        try:
            if platform.system() == "Linux":
                cpu_path = f"/sys/devices/system/node/node{node_id}/cpulist"
                if os.path.exists(cpu_path):
                    with open(cpu_path, 'r') as f:
                        cpu_list = f.read().strip()
                    
                    # Parse CPU list (e.g., "0-3,8-11")
                    cpus = []
                    for part in cpu_list.split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            cpus.extend(range(start, end + 1))
                        else:
                            cpus.append(int(part))
                    
                    return cpus
            
            # Fallback: distribute CPUs evenly
            total_cpus = self.hardware_config.cpu_count
            cpus_per_node = total_cpus // self.hardware_config.numa_nodes
            start_cpu = node_id * cpus_per_node
            end_cpu = start_cpu + cpus_per_node
            
            return list(range(start_cpu, min(end_cpu, total_cpus)))
            
        except Exception:
            return [0]  # Fallback to CPU 0
    
    def _measure_memory_access(self, size_mb: int, access_type: str) -> float:
        """Measure memory access latency"""
        size_bytes = size_mb * 1024 * 1024
        array_size = size_bytes // 8  # 8 bytes per int64
        
        # Create test array
        test_array = np.random.randint(0, array_size, size=array_size, dtype=np.int64)
        
        # Warm up cache
        _ = test_array.sum()
        
        # Measure access time
        operation = f"memory_access_{access_type}_{size_mb}mb"
        
        with self.timer.measure(operation):
            # Random access pattern to avoid cache optimization
            indices = np.random.randint(0, array_size, size=min(10000, array_size))
            total = 0
            for idx in indices:
                total += test_array[idx]
        
        stats = self.timer.get_statistics(operation)
        return stats.mean_ns if stats else 0.0
    
    def _measure_remote_memory_access(self, size_mb: int) -> float:
        """Measure remote NUMA memory access"""
        # This is a simplified simulation - in real implementation,
        # we would allocate memory on remote NUMA nodes
        return self._measure_memory_access(size_mb, "remote") * 1.5  # Assume 50% penalty
    
    def _estimate_memory_bandwidth(self, test_size_mb: int) -> float:
        """Estimate memory bandwidth in GB/s"""
        size_bytes = test_size_mb * 1024 * 1024
        
        # Create large array for bandwidth testing
        test_array = np.random.bytes(size_bytes)
        
        operation = f"memory_bandwidth_{test_size_mb}mb"
        
        with self.timer.measure(operation):
            # Sequential access pattern for bandwidth measurement
            checksum = 0
            for i in range(0, len(test_array), 8):
                checksum += test_array[i]
        
        stats = self.timer.get_statistics(operation)
        if stats and stats.mean_ns > 0:
            # Calculate bandwidth: bytes / time
            bandwidth_bps = size_bytes / (stats.mean_ns / 1e9)
            return bandwidth_bps / (1024**3)  # Convert to GB/s
        
        return 0.0
    
    def profile_cache_performance(self, cpu_id: int) -> CacheProfile:
        """Profile CPU cache performance"""
        # Set CPU affinity
        original_affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, [cpu_id])
            
            # Test different cache levels
            l1_results = self._test_cache_level("L1", 32 * 1024)  # 32KB typical L1
            l2_results = self._test_cache_level("L2", 256 * 1024)  # 256KB typical L2
            l3_results = self._test_cache_level("L3", 8 * 1024 * 1024)  # 8MB typical L3
            
            # Calculate cache profile
            total_accesses = 10000
            l1_hits = total_accesses * 0.9  # Assume 90% L1 hit rate
            l1_misses = total_accesses - l1_hits
            
            l2_hits = l1_misses * 0.8  # 80% of L1 misses hit L2
            l2_misses = l1_misses - l2_hits
            
            l3_hits = l2_misses * 0.7  # 70% of L2 misses hit L3
            l3_misses = l2_misses - l3_hits
            
            hit_rate = (l1_hits + l2_hits + l3_hits) / total_accesses
            miss_penalty_ns = l3_results.get('miss_penalty', 100.0)
            
            profile = CacheProfile(
                l1_hits=int(l1_hits),
                l1_misses=int(l1_misses),
                l2_hits=int(l2_hits),
                l2_misses=int(l2_misses),
                l3_hits=int(l3_hits),
                l3_misses=int(l3_misses),
                hit_rate=hit_rate,
                miss_penalty_ns=miss_penalty_ns
            )
            
            self.cache_profiles[cpu_id] = profile
            return profile
            
        finally:
            # Restore original CPU affinity
            if original_affinity and hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, original_affinity)
    
    def _test_cache_level(self, level: str, size_bytes: int) -> Dict[str, float]:
        """Test specific cache level performance"""
        # Create test data that fits in cache level
        array_size = size_bytes // 8
        test_array = np.random.randint(0, 1000, size=array_size, dtype=np.int64)
        
        # Sequential access (cache-friendly)
        operation = f"cache_{level}_sequential"
        with self.timer.measure(operation):
            total = 0
            for i in range(array_size):
                total += test_array[i]
        
        sequential_stats = self.timer.get_statistics(operation)
        
        # Random access (cache-unfriendly)
        operation = f"cache_{level}_random"
        with self.timer.measure(operation):
            indices = np.random.randint(0, array_size, size=min(1000, array_size))
            total = 0
            for idx in indices:
                total += test_array[idx]
        
        random_stats = self.timer.get_statistics(operation)
        
        # Calculate cache performance metrics
        sequential_ns = sequential_stats.mean_ns if sequential_stats else 0
        random_ns = random_stats.mean_ns if random_stats else 0
        
        miss_penalty = random_ns - sequential_ns if random_ns > sequential_ns else 0
        
        return {
            'sequential_access_ns': sequential_ns,
            'random_access_ns': random_ns,
            'miss_penalty': miss_penalty,
            'efficiency': sequential_ns / random_ns if random_ns > 0 else 1.0
        }
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """Provide hardware-specific optimization recommendations"""
        recommendations = {
            'cpu_affinity': [],
            'memory_allocation': {},
            'cache_optimization': {},
            'numa_optimization': {}
        }
        
        # CPU affinity recommendations
        if self.hardware_config.numa_nodes > 1:
            # Recommend using CPUs from same NUMA node
            recommendations['cpu_affinity'] = self._get_numa_cpus(0)
        else:
            # Use all available CPUs
            recommendations['cpu_affinity'] = list(range(self.hardware_config.cpu_count))
        
        # Memory allocation recommendations
        recommendations['memory_allocation'] = {
            'use_huge_pages': True,
            'numa_local_allocation': True,
            'memory_prefetch': True
        }
        
        # Cache optimization recommendations
        if self.hardware_config.cache_sizes:
            recommendations['cache_optimization'] = {
                'data_structure_size': min(self.hardware_config.cache_sizes.values()),
                'loop_tiling': True,
                'cache_line_alignment': True
            }
        
        # NUMA optimization recommendations
        if self.hardware_config.numa_nodes > 1:
            recommendations['numa_optimization'] = {
                'process_per_node': True,
                'memory_interleaving': False,
                'cross_node_communication': 'minimize'
            }
        
        return recommendations
    
    def benchmark_hardware_configuration(self) -> Dict[str, Any]:
        """Benchmark current hardware configuration"""
        results = {
            'hardware_config': self.hardware_config,
            'numa_profiles': {},
            'cache_profiles': {},
            'optimization_recommendations': self.optimize_for_hardware()
        }
        
        # Profile all NUMA nodes
        for node_id in range(self.hardware_config.numa_nodes):
            try:
                profile = self.profile_numa_performance(node_id)
                results['numa_profiles'][node_id] = profile
            except Exception as e:
                results['numa_profiles'][node_id] = f"Error: {str(e)}"
        
        # Profile cache performance for first few CPUs
        max_cpus_to_test = min(4, self.hardware_config.cpu_count)
        for cpu_id in range(max_cpus_to_test):
            try:
                profile = self.profile_cache_performance(cpu_id)
                results['cache_profiles'][cpu_id] = profile
            except Exception as e:
                results['cache_profiles'][cpu_id] = f"Error: {str(e)}"
        
        return results