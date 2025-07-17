"""
Custom Memory Allocators - High-performance memory management for ultra-low latency.
Implements pool, arena, freelist, buddy, and slab allocators with garbage collection optimization.
"""

import threading
import time
import ctypes
import mmap
import os
import gc
from typing import Any, Dict, List, Optional, Union, Tuple, Generic, TypeVar, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import weakref
import numpy as np
import torch
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class AllocatorType(Enum):
    """Types of memory allocators."""
    POOL = "pool"
    ARENA = "arena"
    FREELIST = "freelist"
    BUDDY = "buddy"
    SLAB = "slab"
    STACK = "stack"


@dataclass
class AllocationMetrics:
    """Metrics for memory allocation."""
    allocations_requested: int = 0
    allocations_successful: int = 0
    allocations_failed: int = 0
    deallocations_requested: int = 0
    deallocations_successful: int = 0
    total_allocated_bytes: int = 0
    total_deallocated_bytes: int = 0
    peak_allocated_bytes: int = 0
    current_allocated_bytes: int = 0
    fragmentation_ratio: float = 0.0
    allocation_time_ns: int = 0
    deallocation_time_ns: int = 0
    gc_runs: int = 0
    gc_time_ns: int = 0


class MemoryBlock:
    """Represents a memory block in custom allocators."""
    
    def __init__(self, start: int, size: int, free: bool = True):
        self.start = start
        self.size = size
        self.free = free
        self.next: Optional['MemoryBlock'] = None
        self.prev: Optional['MemoryBlock'] = None
        self.buddy: Optional['MemoryBlock'] = None
        self.split_level: int = 0
        self.allocation_time: float = 0.0
        self.last_access_time: float = 0.0
    
    def __repr__(self) -> str:
        return f"MemoryBlock(start={self.start}, size={self.size}, free={self.free})"


class BaseAllocator(ABC):
    """Abstract base class for custom allocators."""
    
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.metrics = AllocationMetrics()
        self._lock = threading.RLock()
        self._memory_base = None
        self._initialized = False
        
        logger.info(f"Base allocator created", name=name, size=size)
    
    @abstractmethod
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate memory block."""
        pass
    
    @abstractmethod
    def deallocate(self, ptr: int):
        """Deallocate memory block."""
        pass
    
    @abstractmethod
    def get_utilization(self) -> float:
        """Get memory utilization ratio."""
        pass
    
    @abstractmethod
    def get_fragmentation(self) -> float:
        """Get memory fragmentation ratio."""
        pass
    
    def get_metrics(self) -> AllocationMetrics:
        """Get allocator metrics."""
        with self._lock:
            return self.metrics
    
    def reset_metrics(self):
        """Reset allocator metrics."""
        with self._lock:
            self.metrics = AllocationMetrics()
    
    def _update_allocation_metrics(self, size: int, success: bool, time_ns: int):
        """Update allocation metrics."""
        with self._lock:
            self.metrics.allocations_requested += 1
            self.metrics.allocation_time_ns += time_ns
            
            if success:
                self.metrics.allocations_successful += 1
                self.metrics.total_allocated_bytes += size
                self.metrics.current_allocated_bytes += size
                self.metrics.peak_allocated_bytes = max(
                    self.metrics.peak_allocated_bytes,
                    self.metrics.current_allocated_bytes
                )
            else:
                self.metrics.allocations_failed += 1
    
    def _update_deallocation_metrics(self, size: int, success: bool, time_ns: int):
        """Update deallocation metrics."""
        with self._lock:
            self.metrics.deallocations_requested += 1
            self.metrics.deallocation_time_ns += time_ns
            
            if success:
                self.metrics.deallocations_successful += 1
                self.metrics.total_deallocated_bytes += size
                self.metrics.current_allocated_bytes -= size
    
    def _align_size(self, size: int, alignment: int) -> int:
        """Align size to alignment boundary."""
        return (size + alignment - 1) & ~(alignment - 1)


class PoolAllocator(BaseAllocator):
    """
    Pool allocator for fixed-size objects.
    Extremely fast allocation/deallocation for objects of the same size.
    """
    
    def __init__(self, name: str, object_size: int, pool_size: int, alignment: int = 8):
        self.object_size = self._align_size(object_size, alignment)
        self.pool_size = pool_size
        self.alignment = alignment
        
        total_size = self.object_size * pool_size
        super().__init__(name, total_size)
        
        # Initialize pool
        self._free_list: List[int] = []
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        self._initialize_pool()
        
        logger.info(f"Pool allocator created", 
                   name=name, 
                   object_size=self.object_size,
                   pool_size=pool_size,
                   total_size=total_size)
    
    def _initialize_pool(self):
        """Initialize the memory pool."""
        with self._lock:
            # Create free list
            for i in range(self.pool_size):
                offset = i * self.object_size
                self._free_list.append(offset)
            
            self._initialized = True
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate fixed-size object."""
        start_time = time.time_ns()
        
        if size > self.object_size:
            self._update_allocation_metrics(size, False, time.time_ns() - start_time)
            return None
        
        with self._lock:
            if not self._free_list:
                self._update_allocation_metrics(size, False, time.time_ns() - start_time)
                return None
            
            # Get free block
            offset = self._free_list.pop()
            
            # Create allocation record
            block = MemoryBlock(offset, self.object_size, False)
            block.allocation_time = time.time()
            block.last_access_time = time.time()
            
            self._allocated_blocks[offset] = block
            
            self._update_allocation_metrics(self.object_size, True, time.time_ns() - start_time)
            return offset
    
    def deallocate(self, ptr: int):
        """Deallocate object."""
        start_time = time.time_ns()
        
        with self._lock:
            if ptr not in self._allocated_blocks:
                self._update_deallocation_metrics(0, False, time.time_ns() - start_time)
                return
            
            block = self._allocated_blocks.pop(ptr)
            
            # Return to free list
            self._free_list.append(ptr)
            
            self._update_deallocation_metrics(block.size, True, time.time_ns() - start_time)
    
    def get_utilization(self) -> float:
        """Get pool utilization."""
        with self._lock:
            used_blocks = self.pool_size - len(self._free_list)
            return used_blocks / self.pool_size
    
    def get_fragmentation(self) -> float:
        """Get fragmentation (always 0 for pool allocator)."""
        return 0.0
    
    def get_available_count(self) -> int:
        """Get number of available blocks."""
        with self._lock:
            return len(self._free_list)
    
    def get_allocated_count(self) -> int:
        """Get number of allocated blocks."""
        with self._lock:
            return len(self._allocated_blocks)


class ArenaAllocator(BaseAllocator):
    """
    Arena allocator for variable-size objects.
    Fast allocation with bump pointer, bulk deallocation.
    """
    
    def __init__(self, name: str, size: int, alignment: int = 8):
        super().__init__(name, size)
        self.alignment = alignment
        
        # Initialize arena
        self._current_offset = 0
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        self._checkpoints: List[int] = []
        
        logger.info(f"Arena allocator created", name=name, size=size)
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate variable-size object."""
        start_time = time.time_ns()
        
        aligned_size = self._align_size(size, alignment)
        
        with self._lock:
            if self._current_offset + aligned_size > self.size:
                self._update_allocation_metrics(size, False, time.time_ns() - start_time)
                return None
            
            # Allocate at current offset
            offset = self._current_offset
            self._current_offset += aligned_size
            
            # Create allocation record
            block = MemoryBlock(offset, aligned_size, False)
            block.allocation_time = time.time()
            block.last_access_time = time.time()
            
            self._allocated_blocks[offset] = block
            
            self._update_allocation_metrics(aligned_size, True, time.time_ns() - start_time)
            return offset
    
    def deallocate(self, ptr: int):
        """Deallocate object (no-op for arena)."""
        start_time = time.time_ns()
        
        with self._lock:
            if ptr in self._allocated_blocks:
                block = self._allocated_blocks[ptr]
                block.free = True
                self._update_deallocation_metrics(block.size, True, time.time_ns() - start_time)
            else:
                self._update_deallocation_metrics(0, False, time.time_ns() - start_time)
    
    def reset(self):
        """Reset arena to initial state."""
        with self._lock:
            self._current_offset = 0
            self._allocated_blocks.clear()
            self._checkpoints.clear()
            
            logger.info(f"Arena allocator reset", name=self.name)
    
    def checkpoint(self) -> int:
        """Create checkpoint for rollback."""
        with self._lock:
            checkpoint_id = len(self._checkpoints)
            self._checkpoints.append(self._current_offset)
            return checkpoint_id
    
    def rollback(self, checkpoint_id: int):
        """Rollback to checkpoint."""
        with self._lock:
            if checkpoint_id < len(self._checkpoints):
                self._current_offset = self._checkpoints[checkpoint_id]
                
                # Remove blocks allocated after checkpoint
                to_remove = []
                for offset, block in self._allocated_blocks.items():
                    if offset >= self._current_offset:
                        to_remove.append(offset)
                
                for offset in to_remove:
                    del self._allocated_blocks[offset]
                
                logger.info(f"Arena rollback", checkpoint_id=checkpoint_id, offset=self._current_offset)
    
    def get_utilization(self) -> float:
        """Get arena utilization."""
        with self._lock:
            return self._current_offset / self.size
    
    def get_fragmentation(self) -> float:
        """Get fragmentation (always 0 for arena)."""
        return 0.0


class FreeListAllocator(BaseAllocator):
    """
    Free list allocator for variable-size objects.
    Maintains list of free blocks, supports coalescing.
    """
    
    def __init__(self, name: str, size: int, alignment: int = 8):
        super().__init__(name, size)
        self.alignment = alignment
        
        # Initialize with single free block
        self._free_blocks: List[MemoryBlock] = []
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        self._initialize_free_list()
        
        logger.info(f"FreeList allocator created", name=name, size=size)
    
    def _initialize_free_list(self):
        """Initialize free list with single block."""
        with self._lock:
            initial_block = MemoryBlock(0, self.size, True)
            self._free_blocks.append(initial_block)
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate variable-size object."""
        start_time = time.time_ns()
        
        aligned_size = self._align_size(size, alignment)
        
        with self._lock:
            # Find suitable free block
            for i, block in enumerate(self._free_blocks):
                if block.size >= aligned_size:
                    # Found suitable block
                    self._free_blocks.pop(i)
                    
                    # Split block if too large
                    if block.size > aligned_size:
                        remainder = MemoryBlock(
                            block.start + aligned_size,
                            block.size - aligned_size,
                            True
                        )
                        self._free_blocks.append(remainder)
                    
                    # Create allocated block
                    allocated_block = MemoryBlock(block.start, aligned_size, False)
                    allocated_block.allocation_time = time.time()
                    allocated_block.last_access_time = time.time()
                    
                    self._allocated_blocks[block.start] = allocated_block
                    
                    self._update_allocation_metrics(aligned_size, True, time.time_ns() - start_time)
                    return block.start
            
            # No suitable block found
            self._update_allocation_metrics(size, False, time.time_ns() - start_time)
            return None
    
    def deallocate(self, ptr: int):
        """Deallocate object."""
        start_time = time.time_ns()
        
        with self._lock:
            if ptr not in self._allocated_blocks:
                self._update_deallocation_metrics(0, False, time.time_ns() - start_time)
                return
            
            block = self._allocated_blocks.pop(ptr)
            
            # Create free block
            free_block = MemoryBlock(block.start, block.size, True)
            
            # Insert and coalesce
            self._insert_and_coalesce(free_block)
            
            self._update_deallocation_metrics(block.size, True, time.time_ns() - start_time)
    
    def _insert_and_coalesce(self, block: MemoryBlock):
        """Insert block into free list and coalesce adjacent blocks."""
        # Find insertion point
        insert_index = 0
        for i, free_block in enumerate(self._free_blocks):
            if free_block.start > block.start:
                insert_index = i
                break
        else:
            insert_index = len(self._free_blocks)
        
        # Insert block
        self._free_blocks.insert(insert_index, block)
        
        # Coalesce with next block
        if insert_index < len(self._free_blocks) - 1:
            next_block = self._free_blocks[insert_index + 1]
            if block.start + block.size == next_block.start:
                # Coalesce
                block.size += next_block.size
                self._free_blocks.pop(insert_index + 1)
        
        # Coalesce with previous block
        if insert_index > 0:
            prev_block = self._free_blocks[insert_index - 1]
            if prev_block.start + prev_block.size == block.start:
                # Coalesce
                prev_block.size += block.size
                self._free_blocks.pop(insert_index)
    
    def get_utilization(self) -> float:
        """Get utilization."""
        with self._lock:
            free_bytes = sum(block.size for block in self._free_blocks)
            return (self.size - free_bytes) / self.size
    
    def get_fragmentation(self) -> float:
        """Get fragmentation ratio."""
        with self._lock:
            if not self._free_blocks:
                return 0.0
            
            total_free = sum(block.size for block in self._free_blocks)
            largest_free = max(block.size for block in self._free_blocks)
            
            if total_free == 0:
                return 0.0
            
            return 1.0 - (largest_free / total_free)
    
    def get_free_block_count(self) -> int:
        """Get number of free blocks."""
        with self._lock:
            return len(self._free_blocks)


class BuddyAllocator(BaseAllocator):
    """
    Buddy allocator for power-of-2 sized objects.
    Efficient coalescing and minimal fragmentation.
    """
    
    def __init__(self, name: str, size: int, min_block_size: int = 64):
        # Ensure size is power of 2
        if size & (size - 1) != 0:
            size = 1 << (size.bit_length())
        
        super().__init__(name, size)
        
        self.min_block_size = min_block_size
        self.max_order = (size // min_block_size).bit_length() - 1
        
        # Initialize buddy system
        self._free_lists: Dict[int, List[MemoryBlock]] = {}
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        self._initialize_buddy_system()
        
        logger.info(f"Buddy allocator created", 
                   name=name, 
                   size=size, 
                   min_block_size=min_block_size,
                   max_order=self.max_order)
    
    def _initialize_buddy_system(self):
        """Initialize buddy system with single large block."""
        with self._lock:
            # Initialize free lists
            for order in range(self.max_order + 1):
                self._free_lists[order] = []
            
            # Create initial block
            initial_block = MemoryBlock(0, self.size, True)
            initial_block.split_level = self.max_order
            self._free_lists[self.max_order].append(initial_block)
    
    def _get_order(self, size: int) -> int:
        """Get order for size."""
        if size <= self.min_block_size:
            return 0
        
        # Find smallest power of 2 >= size
        order = 0
        block_size = self.min_block_size
        while block_size < size:
            block_size <<= 1
            order += 1
        
        return min(order, self.max_order)
    
    def _get_block_size(self, order: int) -> int:
        """Get block size for order."""
        return self.min_block_size << order
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate power-of-2 sized object."""
        start_time = time.time_ns()
        
        order = self._get_order(size)
        block_size = self._get_block_size(order)
        
        with self._lock:
            # Find block of appropriate size
            block = self._find_block(order)
            
            if block is None:
                self._update_allocation_metrics(size, False, time.time_ns() - start_time)
                return None
            
            # Mark as allocated
            block.free = False
            block.allocation_time = time.time()
            block.last_access_time = time.time()
            
            self._allocated_blocks[block.start] = block
            
            self._update_allocation_metrics(block_size, True, time.time_ns() - start_time)
            return block.start
    
    def _find_block(self, order: int) -> Optional[MemoryBlock]:
        """Find block of given order."""
        # Try to find block of exact size
        if self._free_lists[order]:
            return self._free_lists[order].pop()
        
        # Try to find larger block and split
        for higher_order in range(order + 1, self.max_order + 1):
            if self._free_lists[higher_order]:
                block = self._free_lists[higher_order].pop()
                return self._split_block(block, order)
        
        return None
    
    def _split_block(self, block: MemoryBlock, target_order: int) -> MemoryBlock:
        """Split block to target order."""
        while block.split_level > target_order:
            # Split block
            block.split_level -= 1
            buddy_size = self._get_block_size(block.split_level)
            
            # Create buddy block
            buddy = MemoryBlock(block.start + buddy_size, buddy_size, True)
            buddy.split_level = block.split_level
            buddy.buddy = block
            block.buddy = buddy
            
            # Add buddy to free list
            self._free_lists[block.split_level].append(buddy)
            
            # Update block size
            block.size = buddy_size
        
        return block
    
    def deallocate(self, ptr: int):
        """Deallocate object."""
        start_time = time.time_ns()
        
        with self._lock:
            if ptr not in self._allocated_blocks:
                self._update_deallocation_metrics(0, False, time.time_ns() - start_time)
                return
            
            block = self._allocated_blocks.pop(ptr)
            
            # Mark as free
            block.free = True
            
            # Try to coalesce with buddy
            self._coalesce_block(block)
            
            self._update_deallocation_metrics(block.size, True, time.time_ns() - start_time)
    
    def _coalesce_block(self, block: MemoryBlock):
        """Coalesce block with buddy."""
        while block.split_level < self.max_order:
            buddy = block.buddy
            
            # Check if buddy exists and is free
            if buddy is None or not buddy.free:
                break
            
            # Find buddy in free list
            if buddy not in self._free_lists[block.split_level]:
                break
            
            # Remove buddy from free list
            self._free_lists[block.split_level].remove(buddy)
            
            # Coalesce
            if block.start > buddy.start:
                block, buddy = buddy, block
            
            block.size += buddy.size
            block.split_level += 1
            
            # Update buddy pointer
            if block.split_level < self.max_order:
                buddy_offset = block.size
                if block.start % (buddy_offset * 2) == 0:
                    buddy_start = block.start + buddy_offset
                else:
                    buddy_start = block.start - buddy_offset
                
                # Find new buddy
                for allocated_block in self._allocated_blocks.values():
                    if allocated_block.start == buddy_start:
                        block.buddy = allocated_block
                        allocated_block.buddy = block
                        break
        
        # Add to appropriate free list
        self._free_lists[block.split_level].append(block)
    
    def get_utilization(self) -> float:
        """Get utilization."""
        with self._lock:
            free_bytes = 0
            for order, blocks in self._free_lists.items():
                block_size = self._get_block_size(order)
                free_bytes += len(blocks) * block_size
            
            return (self.size - free_bytes) / self.size
    
    def get_fragmentation(self) -> float:
        """Get fragmentation ratio."""
        with self._lock:
            # Count free blocks
            free_blocks = sum(len(blocks) for blocks in self._free_lists.values())
            
            if free_blocks <= 1:
                return 0.0
            
            # Internal fragmentation from power-of-2 allocation
            total_requested = 0
            total_allocated = 0
            
            for block in self._allocated_blocks.values():
                # Estimate requested size (this is approximate)
                requested_size = block.size // 2 if block.size > self.min_block_size else block.size
                total_requested += requested_size
                total_allocated += block.size
            
            if total_allocated == 0:
                return 0.0
            
            return 1.0 - (total_requested / total_allocated)


class SlabAllocator(BaseAllocator):
    """
    Slab allocator for objects of specific types.
    Combines multiple pool allocators for different sizes.
    """
    
    def __init__(self, name: str, size_classes: List[int], objects_per_slab: int = 64):
        total_size = sum(size_class * objects_per_slab for size_class in size_classes)
        super().__init__(name, total_size)
        
        self.size_classes = sorted(size_classes)
        self.objects_per_slab = objects_per_slab
        
        # Create pool allocators for each size class
        self._pools: Dict[int, PoolAllocator] = {}
        for size_class in size_classes:
            pool_name = f"{name}_pool_{size_class}"
            self._pools[size_class] = PoolAllocator(
                pool_name, 
                size_class, 
                objects_per_slab
            )
        
        logger.info(f"Slab allocator created", 
                   name=name, 
                   size_classes=size_classes,
                   objects_per_slab=objects_per_slab)
    
    def _find_size_class(self, size: int) -> Optional[int]:
        """Find appropriate size class for size."""
        for size_class in self.size_classes:
            if size <= size_class:
                return size_class
        return None
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate object from appropriate slab."""
        start_time = time.time_ns()
        
        size_class = self._find_size_class(size)
        if size_class is None:
            self._update_allocation_metrics(size, False, time.time_ns() - start_time)
            return None
        
        # Allocate from appropriate pool
        pool = self._pools[size_class]
        offset = pool.allocate(size, alignment)
        
        if offset is not None:
            # Adjust offset to account for pool base
            base_offset = sum(
                sc * self.objects_per_slab 
                for sc in self.size_classes 
                if sc < size_class
            )
            actual_offset = base_offset + offset
            
            self._update_allocation_metrics(size_class, True, time.time_ns() - start_time)
            return actual_offset
        else:
            self._update_allocation_metrics(size, False, time.time_ns() - start_time)
            return None
    
    def deallocate(self, ptr: int):
        """Deallocate object from appropriate slab."""
        start_time = time.time_ns()
        
        # Find which pool owns this pointer
        cumulative_offset = 0
        for size_class in self.size_classes:
            pool_size = size_class * self.objects_per_slab
            
            if ptr < cumulative_offset + pool_size:
                # This pool owns the pointer
                pool = self._pools[size_class]
                pool_offset = ptr - cumulative_offset
                pool.deallocate(pool_offset)
                
                self._update_deallocation_metrics(size_class, True, time.time_ns() - start_time)
                return
            
            cumulative_offset += pool_size
        
        # Pointer not found
        self._update_deallocation_metrics(0, False, time.time_ns() - start_time)
    
    def get_utilization(self) -> float:
        """Get overall utilization."""
        total_capacity = 0
        total_used = 0
        
        for size_class, pool in self._pools.items():
            pool_capacity = pool.pool_size
            pool_used = pool.get_allocated_count()
            
            total_capacity += pool_capacity
            total_used += pool_used
        
        return total_used / total_capacity if total_capacity > 0 else 0.0
    
    def get_fragmentation(self) -> float:
        """Get fragmentation (internal fragmentation from size classes)."""
        # This is complex to calculate accurately
        # Return average utilization of individual pools
        utilizations = [pool.get_utilization() for pool in self._pools.values()]
        return 1.0 - (sum(utilizations) / len(utilizations)) if utilizations else 0.0
    
    def get_pool_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each pool."""
        stats = {}
        for size_class, pool in self._pools.items():
            stats[size_class] = {
                'utilization': pool.get_utilization(),
                'available': pool.get_available_count(),
                'allocated': pool.get_allocated_count(),
                'metrics': pool.get_metrics()
            }
        return stats


class CustomMemoryManager:
    """
    Manager for custom memory allocators.
    Provides unified interface and allocation strategy selection.
    """
    
    def __init__(self):
        self._allocators: Dict[str, BaseAllocator] = {}
        self._allocation_map: Dict[int, Tuple[str, int]] = {}  # ptr -> (allocator_name, size)
        self._default_allocator: Optional[str] = None
        self._lock = threading.RLock()
        
        # Garbage collection settings
        self._gc_enabled = True
        self._gc_threshold = 1000  # Allocations before GC
        self._gc_counter = 0
        
        logger.info("Custom memory manager initialized")
    
    def register_allocator(self, allocator: BaseAllocator, is_default: bool = False):
        """Register an allocator."""
        with self._lock:
            self._allocators[allocator.name] = allocator
            
            if is_default or self._default_allocator is None:
                self._default_allocator = allocator.name
            
            logger.info(f"Allocator registered", name=allocator.name, is_default=is_default)
    
    def allocate(self, size: int, allocator_name: Optional[str] = None, alignment: int = 8) -> Optional[int]:
        """Allocate memory using specified or default allocator."""
        with self._lock:
            if allocator_name is None:
                allocator_name = self._default_allocator
            
            if allocator_name not in self._allocators:
                logger.error(f"Allocator not found", name=allocator_name)
                return None
            
            allocator = self._allocators[allocator_name]
            ptr = allocator.allocate(size, alignment)
            
            if ptr is not None:
                self._allocation_map[ptr] = (allocator_name, size)
                
                # Check garbage collection
                self._gc_counter += 1
                if self._gc_enabled and self._gc_counter >= self._gc_threshold:
                    self._run_gc()
            
            return ptr
    
    def deallocate(self, ptr: int):
        """Deallocate memory."""
        with self._lock:
            if ptr not in self._allocation_map:
                logger.warning(f"Attempting to deallocate unknown pointer", ptr=ptr)
                return
            
            allocator_name, size = self._allocation_map.pop(ptr)
            allocator = self._allocators[allocator_name]
            allocator.deallocate(ptr)
    
    def _run_gc(self):
        """Run garbage collection."""
        start_time = time.time_ns()
        
        try:
            # Python garbage collection
            collected = gc.collect()
            
            # Update metrics for all allocators
            for allocator in self._allocators.values():
                allocator.metrics.gc_runs += 1
                allocator.metrics.gc_time_ns += time.time_ns() - start_time
            
            self._gc_counter = 0
            
            logger.debug(f"Garbage collection completed", objects_collected=collected)
            
        except Exception as e:
            logger.error(f"Garbage collection failed", error=str(e))
    
    def get_allocator(self, name: str) -> Optional[BaseAllocator]:
        """Get allocator by name."""
        with self._lock:
            return self._allocators.get(name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._lock:
            stats = {}
            
            for name, allocator in self._allocators.items():
                stats[name] = {
                    'type': type(allocator).__name__,
                    'size': allocator.size,
                    'utilization': allocator.get_utilization(),
                    'fragmentation': allocator.get_fragmentation(),
                    'metrics': allocator.get_metrics()
                }
            
            stats['manager'] = {
                'active_allocations': len(self._allocation_map),
                'default_allocator': self._default_allocator,
                'gc_enabled': self._gc_enabled,
                'gc_threshold': self._gc_threshold,
                'gc_counter': self._gc_counter
            }
            
            return stats
    
    def optimize_allocators(self):
        """Optimize allocator performance."""
        with self._lock:
            # Force garbage collection
            self._run_gc()
            
            # Analyze allocation patterns
            size_histogram = {}
            for allocator_name, size in self._allocation_map.values():
                if size not in size_histogram:
                    size_histogram[size] = 0
                size_histogram[size] += 1
            
            # Log optimization suggestions
            if size_histogram:
                most_common_size = max(size_histogram.items(), key=lambda x: x[1])
                logger.info(f"Allocation optimization suggestion", 
                           most_common_size=most_common_size[0],
                           frequency=most_common_size[1])
    
    def enable_gc(self, threshold: int = 1000):
        """Enable garbage collection."""
        with self._lock:
            self._gc_enabled = True
            self._gc_threshold = threshold
            logger.info(f"Garbage collection enabled", threshold=threshold)
    
    def disable_gc(self):
        """Disable garbage collection."""
        with self._lock:
            self._gc_enabled = False
            logger.info("Garbage collection disabled")


# Global custom memory manager
_global_custom_memory_manager: Optional[CustomMemoryManager] = None


def get_custom_memory_manager() -> CustomMemoryManager:
    """Get the global custom memory manager."""
    global _global_custom_memory_manager
    if _global_custom_memory_manager is None:
        _global_custom_memory_manager = CustomMemoryManager()
        
        # Register default allocators
        _initialize_default_allocators()
    
    return _global_custom_memory_manager


def _initialize_default_allocators():
    """Initialize default allocators."""
    manager = _global_custom_memory_manager
    
    # Create default allocators
    pool_allocator = PoolAllocator("default_pool", 1024, 1000)
    arena_allocator = ArenaAllocator("default_arena", 1024 * 1024)
    freelist_allocator = FreeListAllocator("default_freelist", 1024 * 1024)
    buddy_allocator = BuddyAllocator("default_buddy", 1024 * 1024)
    
    # Common size classes for slab allocator
    size_classes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    slab_allocator = SlabAllocator("default_slab", size_classes)
    
    # Register allocators
    manager.register_allocator(pool_allocator)
    manager.register_allocator(arena_allocator)
    manager.register_allocator(freelist_allocator, is_default=True)
    manager.register_allocator(buddy_allocator)
    manager.register_allocator(slab_allocator)


def set_custom_memory_manager(manager: CustomMemoryManager):
    """Set the global custom memory manager."""
    global _global_custom_memory_manager
    _global_custom_memory_manager = manager


def get_custom_allocator(name: str) -> Optional[BaseAllocator]:
    """Get custom allocator by name."""
    return get_custom_memory_manager().get_allocator(name)


def set_custom_allocator(allocator: BaseAllocator, is_default: bool = False):
    """Set custom allocator."""
    get_custom_memory_manager().register_allocator(allocator, is_default)


# Tensor allocation functions
def allocate_tensor(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, allocator_name: Optional[str] = None) -> Optional[torch.Tensor]:
    """Allocate tensor using custom allocator."""
    manager = get_custom_memory_manager()
    
    # Calculate size
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_size = int(np.prod(shape)) * element_size
    
    # Allocate memory
    ptr = manager.allocate(total_size, allocator_name)
    
    if ptr is None:
        return None
    
    # Create tensor (this is a simplified implementation)
    # In practice, you'd need to interface with PyTorch's memory allocator
    tensor = torch.empty(shape, dtype=dtype, device=device)
    
    logger.debug(f"Tensor allocated", shape=shape, dtype=str(dtype), device=str(device), ptr=ptr)
    return tensor


def deallocate_tensor(tensor: torch.Tensor):
    """Deallocate tensor using custom allocator."""
    # This would require custom PyTorch memory allocator integration
    # For now, this is a placeholder
    logger.debug(f"Tensor deallocated", shape=tensor.shape)


def benchmark_allocator(allocator: BaseAllocator, num_allocations: int = 10000, size_range: Tuple[int, int] = (64, 1024)) -> Dict[str, Any]:
    """Benchmark allocator performance."""
    import random
    import time
    
    start_time = time.time_ns()
    
    # Allocation phase
    pointers = []
    allocation_times = []
    
    for _ in range(num_allocations):
        size = random.randint(*size_range)
        
        alloc_start = time.time_ns()
        ptr = allocator.allocate(size)
        alloc_end = time.time_ns()
        
        if ptr is not None:
            pointers.append(ptr)
            allocation_times.append(alloc_end - alloc_start)
    
    # Deallocation phase
    deallocation_times = []
    
    for ptr in pointers:
        dealloc_start = time.time_ns()
        allocator.deallocate(ptr)
        dealloc_end = time.time_ns()
        
        deallocation_times.append(dealloc_end - dealloc_start)
    
    end_time = time.time_ns()
    
    # Calculate metrics
    total_time_ns = end_time - start_time
    successful_allocations = len(pointers)
    
    return {
        'allocator_name': allocator.name,
        'allocator_type': type(allocator).__name__,
        'num_allocations_requested': num_allocations,
        'num_allocations_successful': successful_allocations,
        'success_rate': successful_allocations / num_allocations,
        'total_time_ns': total_time_ns,
        'total_time_ms': total_time_ns / 1_000_000,
        'avg_allocation_time_ns': sum(allocation_times) / len(allocation_times) if allocation_times else 0,
        'avg_deallocation_time_ns': sum(deallocation_times) / len(deallocation_times) if deallocation_times else 0,
        'throughput_allocations_per_sec': successful_allocations / (total_time_ns / 1_000_000_000),
        'utilization': allocator.get_utilization(),
        'fragmentation': allocator.get_fragmentation(),
        'metrics': allocator.get_metrics()
    }