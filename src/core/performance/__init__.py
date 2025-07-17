"""
Zero-Copy Performance Framework
High-performance, zero-copy data structures and algorithms for sub-millisecond performance.
"""

from .zero_copy_framework import (
    ZeroCopyBuffer,
    ZeroCopyTensor,
    ZeroCopyArray,
    ZeroCopyBufferPool,
    zero_copy_tensor,
    zero_copy_array,
    zero_copy_context
)

from .memory_mapped_structures import (
    MemoryMappedTensor,
    MemoryMappedArray,
    MemoryMappedQueue,
    MemoryMappedHashMap,
    SharedMemoryManager,
    memory_mapped_tensor,
    memory_mapped_array
)

from .simd_operations import (
    SIMDMath,
    vectorized_add,
    vectorized_mul,
    vectorized_sum,
    vectorized_mean,
    vectorized_std,
    vectorized_max,
    vectorized_min,
    vectorized_softmax,
    vectorized_relu,
    vectorized_tanh,
    vectorized_sigmoid,
    vectorized_matmul,
    enable_simd_optimizations
)

from .lock_free_structures import (
    LockFreeQueue,
    LockFreeStack,
    LockFreeHashMap,
    LockFreeCircularBuffer,
    AtomicCounter,
    SpinLock,
    RCUPointer,
    HazardPointer
)

from .custom_allocators import (
    PoolAllocator,
    ArenaAllocator,
    FreeListAllocator,
    BuddyAllocator,
    SlabAllocator,
    CustomMemoryManager,
    get_custom_allocator,
    set_custom_allocator,
    allocate_tensor,
    deallocate_tensor
)

from .performance_monitor import (
    PerformanceProfiler,
    LatencyTracker,
    ThroughputMonitor,
    MemoryProfiler,
    CPUProfiler,
    GPUProfiler,
    get_performance_monitor,
    profile_function,
    profile_context,
    benchmark_function
)

__all__ = [
    # Zero-copy framework
    'ZeroCopyBuffer',
    'ZeroCopyTensor',
    'ZeroCopyArray',
    'ZeroCopyBufferPool',
    'zero_copy_tensor',
    'zero_copy_array',
    'zero_copy_context',
    
    # Memory-mapped structures
    'MemoryMappedTensor',
    'MemoryMappedArray',
    'MemoryMappedQueue',
    'MemoryMappedHashMap',
    'SharedMemoryManager',
    'memory_mapped_tensor',
    'memory_mapped_array',
    
    # SIMD operations
    'SIMDMath',
    'vectorized_add',
    'vectorized_mul',
    'vectorized_sum',
    'vectorized_mean',
    'vectorized_std',
    'vectorized_max',
    'vectorized_min',
    'vectorized_softmax',
    'vectorized_relu',
    'vectorized_tanh',
    'vectorized_sigmoid',
    'vectorized_matmul',
    'enable_simd_optimizations',
    
    # Lock-free structures
    'LockFreeQueue',
    'LockFreeStack',
    'LockFreeHashMap',
    'LockFreeCircularBuffer',
    'AtomicCounter',
    'SpinLock',
    'RCUPointer',
    'HazardPointer',
    
    # Custom allocators
    'PoolAllocator',
    'ArenaAllocator',
    'FreeListAllocator',
    'BuddyAllocator',
    'SlabAllocator',
    'CustomMemoryManager',
    'get_custom_allocator',
    'set_custom_allocator',
    'allocate_tensor',
    'deallocate_tensor',
    
    # Performance monitoring
    'PerformanceProfiler',
    'LatencyTracker',
    'ThroughputMonitor',
    'MemoryProfiler',
    'CPUProfiler',
    'GPUProfiler',
    'get_performance_monitor',
    'profile_function',
    'profile_context',
    'benchmark_function'
]