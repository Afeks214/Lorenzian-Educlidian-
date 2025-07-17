"""
Memory-Mapped Data Structures - Ultra-fast shared memory operations.
Implements memory-mapped tensors, arrays, queues, and hashmaps for zero-copy inter-process communication.
"""

import logging


import mmap
import os
import numpy as np
import torch
import threading
import time
import struct
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple, Generic, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import tempfile
import pickle
import json
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class MemoryMapMode(Enum):
    """Memory mapping modes."""
    READ_ONLY = mmap.PROT_READ
    READ_WRITE = mmap.PROT_READ | mmap.PROT_WRITE
    WRITE_ONLY = mmap.PROT_WRITE
    SHARED = mmap.MAP_SHARED
    PRIVATE = mmap.MAP_PRIVATE


@dataclass
class MemoryMapInfo:
    """Information about memory-mapped region."""
    filename: str
    size: int
    mode: MemoryMapMode
    offset: int
    created_at: float
    access_count: int = 0
    last_access: float = 0.0


class MemoryMappedTensor:
    """
    Memory-mapped tensor for zero-copy sharing between processes.
    Provides tensor operations on memory-mapped data.
    """
    
    def __init__(self, 
                 shape: Tuple[int, ...], 
                 dtype: torch.dtype = torch.float32,
                 filename: Optional[str] = None,
                 create: bool = True):
        
        self.shape = shape
        self.dtype = dtype
        self.numel = int(np.prod(shape))
        
        # Calculate size in bytes
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.size_bytes = self.numel * self.element_size
        
        # Create or open memory-mapped file
        if filename is None:
            self.filename = tempfile.mktemp(suffix='.mmap')
        else:
            self.filename = filename
        
        self.create = create
        self._mmap = None
        self._tensor = None
        self._lock = threading.RLock()
        
        # Initialize memory mapping
        self._init_memory_mapping()
        
        logger.debug("Memory-mapped tensor created", 
                    shape=shape, 
                    dtype=str(dtype),
                    filename=self.filename,
                    size_bytes=self.size_bytes)
    
    def _init_memory_mapping(self):
        """Initialize memory mapping."""
        with self._lock:
            if self.create:
                # Create new file
                with open(self.filename, 'wb') as f:
                    # Write header with metadata
                    header = struct.pack('Q' * len(self.shape), *self.shape)
                    f.write(header)
                    
                    # Write zeros for tensor data
                    f.write(b'\x00' * self.size_bytes)
                
                # Open for read/write
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
                
            else:
                # Open existing file
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
                
                # Read header to verify shape
                header_size = len(self.shape) * 8  # 8 bytes per dimension
                header_data = self._mmap[:header_size]
                file_shape = struct.unpack('Q' * len(self.shape), header_data)
                
                if file_shape != self.shape:
                    raise ValueError(f"Shape mismatch: expected {self.shape}, got {file_shape}")
            
            # Create tensor view of memory-mapped data
            self._create_tensor_view()
    
    def _create_tensor_view(self):
        """Create tensor view of memory-mapped data."""
        header_size = len(self.shape) * 8
        data_offset = header_size
        
        # Create numpy array view
        if self.dtype == torch.float32:
            np_dtype = np.float32
        elif self.dtype == torch.float64:
            np_dtype = np.float64
        elif self.dtype == torch.int32:
            np_dtype = np.int32
        elif self.dtype == torch.int64:
            np_dtype = np.int64
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        # Create numpy array from memory map
        np_array = np.frombuffer(
            self._mmap, 
            dtype=np_dtype, 
            count=self.numel,
            offset=data_offset
        ).reshape(self.shape)
        
        # Create tensor from numpy array
        self._tensor = torch.from_numpy(np_array)
        
        logger.debug("Tensor view created", shape=self.shape, dtype=str(self.dtype))
    
    def tensor(self) -> torch.Tensor:
        """Get the underlying tensor."""
        return self._tensor
    
    def numpy(self) -> np.ndarray:
        """Get numpy array view."""
        return self._tensor.numpy()
    
    def copy_from(self, source: Union[torch.Tensor, np.ndarray]):
        """Copy data from source tensor/array."""
        with self._lock:
            if isinstance(source, torch.Tensor):
                source_data = source.detach().cpu()
            else:
                source_data = torch.from_numpy(source)
            
            if source_data.shape != self.shape:
                raise ValueError(f"Shape mismatch: {source_data.shape} vs {self.shape}")
            
            self._tensor.copy_(source_data)
            self._mmap.flush()
    
    def copy_to(self, target: Union[torch.Tensor, np.ndarray]):
        """Copy data to target tensor/array."""
        with self._lock:
            if isinstance(target, torch.Tensor):
                target.copy_(self._tensor)
            else:
                target[:] = self._tensor.numpy()
    
    def zero_(self):
        """Zero out the tensor."""
        with self._lock:
            self._tensor.zero_()
            self._mmap.flush()
    
    def fill_(self, value: float):
        """Fill tensor with value."""
        with self._lock:
            self._tensor.fill_(value)
            self._mmap.flush()
    
    def sync(self):
        """Synchronize memory-mapped data to disk."""
        with self._lock:
            self._mmap.flush()
    
    def close(self):
        """Close memory mapping."""
        with self._lock:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if hasattr(self, '_file') and self._file:
                self._file.close()
                self._file = None
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
    
    def __getitem__(self, key):
        """Support indexing."""
        return self._tensor[key]
    
    def __setitem__(self, key, value):
        """Support assignment."""
        with self._lock:
            self._tensor[key] = value
            self._mmap.flush()
    
    def __repr__(self) -> str:
        return f"MemoryMappedTensor(shape={self.shape}, dtype={self.dtype}, file={self.filename})"


class MemoryMappedArray:
    """
    Memory-mapped numpy array for zero-copy sharing.
    Provides array operations on memory-mapped data.
    """
    
    def __init__(self, 
                 shape: Tuple[int, ...], 
                 dtype: np.dtype = np.float32,
                 filename: Optional[str] = None,
                 create: bool = True):
        
        self.shape = shape
        self.dtype = dtype
        self.size_bytes = int(np.prod(shape)) * dtype.itemsize
        
        if filename is None:
            self.filename = tempfile.mktemp(suffix='.mmap')
        else:
            self.filename = filename
        
        self.create = create
        self._array = None
        self._lock = threading.RLock()
        
        # Initialize memory mapping
        self._init_memory_mapping()
        
        logger.debug("Memory-mapped array created", 
                    shape=shape, 
                    dtype=str(dtype),
                    filename=self.filename,
                    size_bytes=self.size_bytes)
    
    def _init_memory_mapping(self):
        """Initialize memory mapping."""
        with self._lock:
            if self.create:
                # Create new memory-mapped array
                self._array = np.memmap(
                    self.filename, 
                    dtype=self.dtype, 
                    mode='w+', 
                    shape=self.shape
                )
            else:
                # Open existing memory-mapped array
                self._array = np.memmap(
                    self.filename, 
                    dtype=self.dtype, 
                    mode='r+', 
                    shape=self.shape
                )
    
    def array(self) -> np.ndarray:
        """Get the underlying array."""
        return self._array
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor."""
        tensor = torch.from_numpy(self._array)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def copy_from(self, source: Union[torch.Tensor, np.ndarray]):
        """Copy data from source."""
        with self._lock:
            if isinstance(source, torch.Tensor):
                source_data = source.detach().cpu().numpy()
            else:
                source_data = source
            
            if source_data.shape != self.shape:
                raise ValueError(f"Shape mismatch: {source_data.shape} vs {self.shape}")
            
            self._array[:] = source_data
            self._array.flush()
    
    def copy_to(self, target: Union[torch.Tensor, np.ndarray]):
        """Copy data to target."""
        with self._lock:
            if isinstance(target, torch.Tensor):
                target.copy_(torch.from_numpy(self._array))
            else:
                target[:] = self._array
    
    def zero_(self):
        """Zero out the array."""
        with self._lock:
            self._array.fill(0)
            self._array.flush()
    
    def fill_(self, value: float):
        """Fill array with value."""
        with self._lock:
            self._array.fill(value)
            self._array.flush()
    
    def sync(self):
        """Synchronize to disk."""
        with self._lock:
            self._array.flush()
    
    def close(self):
        """Close memory mapping."""
        with self._lock:
            if self._array is not None:
                del self._array
                self._array = None
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
    
    def __getitem__(self, key):
        """Support indexing."""
        return self._array[key]
    
    def __setitem__(self, key, value):
        """Support assignment."""
        with self._lock:
            self._array[key] = value
            self._array.flush()
    
    def __repr__(self) -> str:
        return f"MemoryMappedArray(shape={self.shape}, dtype={self.dtype}, file={self.filename})"


class MemoryMappedQueue:
    """
    Memory-mapped queue for zero-copy inter-process communication.
    Implements a circular buffer with atomic operations.
    """
    
    def __init__(self, 
                 capacity: int, 
                 item_size: int,
                 filename: Optional[str] = None,
                 create: bool = True):
        
        self.capacity = capacity
        self.item_size = item_size
        
        # Calculate total size: header + data
        self.header_size = 64  # head, tail, size, etc.
        self.data_size = capacity * item_size
        self.total_size = self.header_size + self.data_size
        
        if filename is None:
            self.filename = tempfile.mktemp(suffix='.queue')
        else:
            self.filename = filename
        
        self.create = create
        self._mmap = None
        self._lock = threading.RLock()
        
        # Initialize memory mapping
        self._init_memory_mapping()
        
        logger.debug("Memory-mapped queue created", 
                    capacity=capacity, 
                    item_size=item_size,
                    filename=self.filename,
                    total_size=self.total_size)
    
    def _init_memory_mapping(self):
        """Initialize memory mapping."""
        with self._lock:
            if self.create:
                # Create new file
                with open(self.filename, 'wb') as f:
                    f.write(b'\x00' * self.total_size)
                
                # Open for read/write
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
                
                # Initialize header
                self._set_head(0)
                self._set_tail(0)
                self._set_size(0)
                
            else:
                # Open existing file
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
    
    def _get_head(self) -> int:
        """Get head position."""
        return struct.unpack('Q', self._mmap[0:8])[0]
    
    def _set_head(self, value: int):
        """Set head position."""
        self._mmap[0:8] = struct.pack('Q', value)
    
    def _get_tail(self) -> int:
        """Get tail position."""
        return struct.unpack('Q', self._mmap[8:16])[0]
    
    def _set_tail(self, value: int):
        """Set tail position."""
        self._mmap[8:16] = struct.pack('Q', value)
    
    def _get_size(self) -> int:
        """Get current size."""
        return struct.unpack('Q', self._mmap[16:24])[0]
    
    def _set_size(self, value: int):
        """Set current size."""
        self._mmap[16:24] = struct.pack('Q', value)
    
    def _get_data_offset(self, index: int) -> int:
        """Get data offset for index."""
        return self.header_size + (index * self.item_size)
    
    def put(self, data: bytes) -> bool:
        """Put data in queue."""
        if len(data) > self.item_size:
            raise ValueError(f"Data too large: {len(data)} > {self.item_size}")
        
        with self._lock:
            current_size = self._get_size()
            if current_size >= self.capacity:
                return False  # Queue full
            
            tail = self._get_tail()
            offset = self._get_data_offset(tail)
            
            # Write data
            padded_data = data + b'\x00' * (self.item_size - len(data))
            self._mmap[offset:offset + self.item_size] = padded_data
            
            # Update tail and size
            self._set_tail((tail + 1) % self.capacity)
            self._set_size(current_size + 1)
            
            self._mmap.flush()
            return True
    
    def get(self) -> Optional[bytes]:
        """Get data from queue."""
        with self._lock:
            current_size = self._get_size()
            if current_size == 0:
                return None  # Queue empty
            
            head = self._get_head()
            offset = self._get_data_offset(head)
            
            # Read data
            data = self._mmap[offset:offset + self.item_size]
            
            # Update head and size
            self._set_head((head + 1) % self.capacity)
            self._set_size(current_size - 1)
            
            # Remove null padding
            return data.rstrip(b'\x00')
    
    def size(self) -> int:
        """Get current queue size."""
        return self._get_size()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.size() >= self.capacity
    
    def clear(self):
        """Clear the queue."""
        with self._lock:
            self._set_head(0)
            self._set_tail(0)
            self._set_size(0)
            self._mmap.flush()
    
    def sync(self):
        """Synchronize to disk."""
        with self._lock:
            self._mmap.flush()
    
    def close(self):
        """Close memory mapping."""
        with self._lock:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if hasattr(self, '_file') and self._file:
                self._file.close()
                self._file = None
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
    
    def __len__(self) -> int:
        return self.size()
    
    def __repr__(self) -> str:
        return f"MemoryMappedQueue(capacity={self.capacity}, size={self.size()}, file={self.filename})"


class MemoryMappedHashMap:
    """
    Memory-mapped hash map for zero-copy key-value storage.
    Implements open addressing with linear probing.
    """
    
    def __init__(self, 
                 capacity: int, 
                 key_size: int,
                 value_size: int,
                 filename: Optional[str] = None,
                 create: bool = True):
        
        self.capacity = capacity
        self.key_size = key_size
        self.value_size = value_size
        
        # Calculate sizes
        self.entry_size = key_size + value_size + 1  # +1 for occupied flag
        self.header_size = 64  # size, count, etc.
        self.data_size = capacity * self.entry_size
        self.total_size = self.header_size + self.data_size
        
        if filename is None:
            self.filename = tempfile.mktemp(suffix='.hashmap')
        else:
            self.filename = filename
        
        self.create = create
        self._mmap = None
        self._lock = threading.RLock()
        
        # Initialize memory mapping
        self._init_memory_mapping()
        
        logger.debug("Memory-mapped hashmap created", 
                    capacity=capacity, 
                    key_size=key_size,
                    value_size=value_size,
                    filename=self.filename,
                    total_size=self.total_size)
    
    def _init_memory_mapping(self):
        """Initialize memory mapping."""
        with self._lock:
            if self.create:
                # Create new file
                with open(self.filename, 'wb') as f:
                    f.write(b'\x00' * self.total_size)
                
                # Open for read/write
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
                
                # Initialize header
                self._set_count(0)
                
            else:
                # Open existing file
                self._file = open(self.filename, 'r+b')
                self._mmap = mmap.mmap(self._file.fileno(), 0)
    
    def _get_count(self) -> int:
        """Get current count."""
        return struct.unpack('Q', self._mmap[0:8])[0]
    
    def _set_count(self, value: int):
        """Set current count."""
        self._mmap[0:8] = struct.pack('Q', value)
    
    def _hash_key(self, key: bytes) -> int:
        """Hash key to index."""
        return hash(key) % self.capacity
    
    def _get_entry_offset(self, index: int) -> int:
        """Get entry offset for index."""
        return self.header_size + (index * self.entry_size)
    
    def _is_occupied(self, index: int) -> bool:
        """Check if entry is occupied."""
        offset = self._get_entry_offset(index)
        return self._mmap[offset] == 1
    
    def _set_occupied(self, index: int, occupied: bool):
        """Set entry occupied status."""
        offset = self._get_entry_offset(index)
        self._mmap[offset] = 1 if occupied else 0
    
    def _get_key(self, index: int) -> bytes:
        """Get key at index."""
        offset = self._get_entry_offset(index) + 1  # +1 for occupied flag
        return self._mmap[offset:offset + self.key_size]
    
    def _set_key(self, index: int, key: bytes):
        """Set key at index."""
        if len(key) > self.key_size:
            raise ValueError(f"Key too large: {len(key)} > {self.key_size}")
        
        offset = self._get_entry_offset(index) + 1
        padded_key = key + b'\x00' * (self.key_size - len(key))
        self._mmap[offset:offset + self.key_size] = padded_key
    
    def _get_value(self, index: int) -> bytes:
        """Get value at index."""
        offset = self._get_entry_offset(index) + 1 + self.key_size
        return self._mmap[offset:offset + self.value_size]
    
    def _set_value(self, index: int, value: bytes):
        """Set value at index."""
        if len(value) > self.value_size:
            raise ValueError(f"Value too large: {len(value)} > {self.value_size}")
        
        offset = self._get_entry_offset(index) + 1 + self.key_size
        padded_value = value + b'\x00' * (self.value_size - len(value))
        self._mmap[offset:offset + self.value_size] = padded_value
    
    def _find_slot(self, key: bytes) -> Tuple[int, bool]:
        """Find slot for key. Returns (index, found)."""
        start_index = self._hash_key(key)
        
        for i in range(self.capacity):
            index = (start_index + i) % self.capacity
            
            if not self._is_occupied(index):
                return index, False  # Empty slot found
            
            if self._get_key(index).rstrip(b'\x00') == key:
                return index, True  # Key found
        
        raise RuntimeError("HashMap is full")
    
    def put(self, key: bytes, value: bytes) -> bool:
        """Put key-value pair."""
        with self._lock:
            try:
                index, found = self._find_slot(key)
                
                if not found:
                    # New entry
                    current_count = self._get_count()
                    if current_count >= self.capacity:
                        return False  # HashMap full
                    
                    self._set_occupied(index, True)
                    self._set_count(current_count + 1)
                
                self._set_key(index, key)
                self._set_value(index, value)
                
                self._mmap.flush()
                return True
                
            except RuntimeError:
                return False  # HashMap full
    
    def get(self, key: bytes) -> Optional[bytes]:
        """Get value for key."""
        with self._lock:
            try:
                index, found = self._find_slot(key)
                
                if found:
                    value = self._get_value(index)
                    return value.rstrip(b'\x00')
                
                return None
                
            except RuntimeError:
                return None
    
    def remove(self, key: bytes) -> bool:
        """Remove key-value pair."""
        with self._lock:
            try:
                index, found = self._find_slot(key)
                
                if found:
                    self._set_occupied(index, False)
                    current_count = self._get_count()
                    self._set_count(current_count - 1)
                    
                    self._mmap.flush()
                    return True
                
                return False
                
            except RuntimeError:
                return False
    
    def contains(self, key: bytes) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def size(self) -> int:
        """Get current size."""
        return self._get_count()
    
    def is_empty(self) -> bool:
        """Check if hashmap is empty."""
        return self.size() == 0
    
    def clear(self):
        """Clear the hashmap."""
        with self._lock:
            # Clear all entries
            for i in range(self.capacity):
                self._set_occupied(i, False)
            
            self._set_count(0)
            self._mmap.flush()
    
    def sync(self):
        """Synchronize to disk."""
        with self._lock:
            self._mmap.flush()
    
    def close(self):
        """Close memory mapping."""
        with self._lock:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if hasattr(self, '_file') and self._file:
                self._file.close()
                self._file = None
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, key: bytes) -> bool:
        return self.contains(key)
    
    def __getitem__(self, key: bytes) -> bytes:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: bytes, value: bytes):
        if not self.put(key, value):
            raise RuntimeError("HashMap is full")
    
    def __delitem__(self, key: bytes):
        if not self.remove(key):
            raise KeyError(key)
    
    def __repr__(self) -> str:
        return f"MemoryMappedHashMap(capacity={self.capacity}, size={self.size()}, file={self.filename})"


class SharedMemoryManager:
    """
    Manager for shared memory-mapped structures.
    Provides centralized management of memory-mapped files.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            base_path = tempfile.gettempdir()
        
        self.base_path = base_path
        self._structures: Dict[str, Any] = {}
        self._info: Dict[str, MemoryMapInfo] = {}
        self._lock = threading.RLock()
        
        logger.info("Shared memory manager initialized", base_path=base_path)
    
    def create_tensor(self, 
                     name: str, 
                     shape: Tuple[int, ...], 
                     dtype: torch.dtype = torch.float32) -> MemoryMappedTensor:
        """Create a memory-mapped tensor."""
        with self._lock:
            if name in self._structures:
                raise ValueError(f"Structure '{name}' already exists")
            
            filename = os.path.join(self.base_path, f"{name}.tensor")
            tensor = MemoryMappedTensor(shape, dtype, filename, create=True)
            
            self._structures[name] = tensor
            self._info[name] = MemoryMapInfo(
                filename=filename,
                size=tensor.size_bytes,
                mode=MemoryMapMode.READ_WRITE,
                offset=0,
                created_at=time.time()
            )
            
            logger.info("Memory-mapped tensor created", name=name, shape=shape)
            return tensor
    
    def create_array(self, 
                    name: str, 
                    shape: Tuple[int, ...], 
                    dtype: np.dtype = np.float32) -> MemoryMappedArray:
        """Create a memory-mapped array."""
        with self._lock:
            if name in self._structures:
                raise ValueError(f"Structure '{name}' already exists")
            
            filename = os.path.join(self.base_path, f"{name}.array")
            array = MemoryMappedArray(shape, dtype, filename, create=True)
            
            self._structures[name] = array
            self._info[name] = MemoryMapInfo(
                filename=filename,
                size=array.size_bytes,
                mode=MemoryMapMode.READ_WRITE,
                offset=0,
                created_at=time.time()
            )
            
            logger.info("Memory-mapped array created", name=name, shape=shape)
            return array
    
    def create_queue(self, 
                    name: str, 
                    capacity: int, 
                    item_size: int) -> MemoryMappedQueue:
        """Create a memory-mapped queue."""
        with self._lock:
            if name in self._structures:
                raise ValueError(f"Structure '{name}' already exists")
            
            filename = os.path.join(self.base_path, f"{name}.queue")
            queue = MemoryMappedQueue(capacity, item_size, filename, create=True)
            
            self._structures[name] = queue
            self._info[name] = MemoryMapInfo(
                filename=filename,
                size=queue.total_size,
                mode=MemoryMapMode.READ_WRITE,
                offset=0,
                created_at=time.time()
            )
            
            logger.info("Memory-mapped queue created", name=name, capacity=capacity)
            return queue
    
    def create_hashmap(self, 
                      name: str, 
                      capacity: int, 
                      key_size: int,
                      value_size: int) -> MemoryMappedHashMap:
        """Create a memory-mapped hashmap."""
        with self._lock:
            if name in self._structures:
                raise ValueError(f"Structure '{name}' already exists")
            
            filename = os.path.join(self.base_path, f"{name}.hashmap")
            hashmap = MemoryMappedHashMap(capacity, key_size, value_size, filename, create=True)
            
            self._structures[name] = hashmap
            self._info[name] = MemoryMapInfo(
                filename=filename,
                size=hashmap.total_size,
                mode=MemoryMapMode.READ_WRITE,
                offset=0,
                created_at=time.time()
            )
            
            logger.info("Memory-mapped hashmap created", name=name, capacity=capacity)
            return hashmap
    
    def get_structure(self, name: str) -> Optional[Any]:
        """Get structure by name."""
        with self._lock:
            if name in self._structures:
                # Update access info
                self._info[name].access_count += 1
                self._info[name].last_access = time.time()
                
                return self._structures[name]
            return None
    
    def remove_structure(self, name: str) -> bool:
        """Remove structure."""
        with self._lock:
            if name not in self._structures:
                return False
            
            structure = self._structures[name]
            info = self._info[name]
            
            # Close and cleanup
            structure.close()
            
            # Remove file
            try:
                os.remove(info.filename)
            except OSError:
                pass
            
            # Remove from tracking
            del self._structures[name]
            del self._info[name]
            
            logger.info("Memory-mapped structure removed", name=name)
            return True
    
    def list_structures(self) -> List[str]:
        """List all structure names."""
        with self._lock:
            return list(self._structures.keys())
    
    def get_info(self, name: str) -> Optional[MemoryMapInfo]:
        """Get structure info."""
        with self._lock:
            return self._info.get(name)
    
    def get_total_size(self) -> int:
        """Get total size of all structures."""
        with self._lock:
            return sum(info.size for info in self._info.values())
    
    def sync_all(self):
        """Synchronize all structures."""
        with self._lock:
            for structure in self._structures.values():
                structure.sync()
    
    def close_all(self):
        """Close all structures."""
        with self._lock:
            for structure in self._structures.values():
                structure.close()
            
            self._structures.clear()
            self._info.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close_all()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')


# Global shared memory manager
_global_shared_memory_manager: Optional[SharedMemoryManager] = None


def get_shared_memory_manager() -> SharedMemoryManager:
    """Get the global shared memory manager."""
    global _global_shared_memory_manager
    if _global_shared_memory_manager is None:
        _global_shared_memory_manager = SharedMemoryManager()
    return _global_shared_memory_manager


def set_shared_memory_manager(manager: SharedMemoryManager):
    """Set the global shared memory manager."""
    global _global_shared_memory_manager
    if _global_shared_memory_manager is not None:
        _global_shared_memory_manager.close_all()
    _global_shared_memory_manager = manager


# Convenience functions
def memory_mapped_tensor(name: str, 
                        shape: Tuple[int, ...], 
                        dtype: torch.dtype = torch.float32) -> MemoryMappedTensor:
    """Create a memory-mapped tensor."""
    return get_shared_memory_manager().create_tensor(name, shape, dtype)


def memory_mapped_array(name: str, 
                       shape: Tuple[int, ...], 
                       dtype: np.dtype = np.float32) -> MemoryMappedArray:
    """Create a memory-mapped array."""
    return get_shared_memory_manager().create_array(name, shape, dtype)


@contextmanager
def memory_mapped_context(name: str):
    """Context manager for memory-mapped structures."""
    manager = get_shared_memory_manager()
    try:
        yield manager.get_structure(name)
    finally:
        # Sync on exit
        structure = manager.get_structure(name)
        if structure and hasattr(structure, 'sync'):
            structure.sync()