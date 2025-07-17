"""
Optimized data storage with compression and efficient retrieval

This module implements high-performance data storage systems with advanced
compression algorithms and optimized retrieval for financial market data.
"""

import numpy as np
import pandas as pd
import time
import threading
import pickle
import lz4.frame
import zstandard as zstd
import blosc
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import os
import struct
import mmap
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
from collections import deque
import psutil
from contextlib import contextmanager
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CompressionType(Enum):
    """Compression algorithms supported"""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BLOSC = "blosc"
    GZIP = "gzip"
    BROTLI = "brotli"

class StorageFormat(Enum):
    """Storage formats supported"""
    PICKLE = "pickle"
    PARQUET = "parquet"
    FEATHER = "feather"
    HDF5 = "hdf5"
    NUMPY = "numpy"
    BINARY = "binary"

@dataclass
class StorageMetrics:
    """Storage performance metrics"""
    total_writes: int = 0
    total_reads: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    avg_write_latency_us: float = 0.0
    avg_read_latency_us: float = 0.0
    compression_ratio: float = 0.0
    storage_utilization_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class DataBlock:
    """Compressed data block"""
    data: bytes
    metadata: Dict[str, Any]
    compression_type: CompressionType
    original_size: int
    compressed_size: int
    checksum: str
    timestamp: float = field(default_factory=time.time)

class CompressedDataStore:
    """High-performance compressed data storage"""
    
    def __init__(self, 
                 storage_path: str,
                 compression_type: CompressionType = CompressionType.ZSTD,
                 compression_level: int = 3,
                 block_size: int = 1024 * 1024,  # 1MB blocks
                 enable_checksum: bool = True,
                 enable_indexing: bool = True):
        
        self.storage_path = Path(storage_path)
        self.compression_type = compression_type
        self.compression_level = compression_level
        self.block_size = block_size
        self.enable_checksum = enable_checksum
        self.enable_indexing = enable_indexing
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize compressor
        self.compressor = self._create_compressor()
        
        # Block index for fast retrieval
        self.block_index = {} if enable_indexing else None
        self.index_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = StorageMetrics()
        self.metrics_lock = threading.Lock()
        
        # Write buffer for batching
        self.write_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_size_limit = 100 * 1024 * 1024  # 100MB cache
        
        # Load existing index
        self._load_index()
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.flush_buffer()
        self._save_index()
        logger.info("CompressedDataStore cleanup completed")
    
    def _create_compressor(self):
        """Create compressor based on type"""
        if self.compression_type == CompressionType.ZSTD:
            return zstd.ZstdCompressor(level=self.compression_level)
        elif self.compression_type == CompressionType.LZ4:
            return None  # LZ4 doesn't need persistent compressor
        elif self.compression_type == CompressionType.BLOSC:
            return None  # Blosc doesn't need persistent compressor
        else:
            return None
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using selected algorithm"""
        if self.compression_type == CompressionType.NONE:
            return data
        elif self.compression_type == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif self.compression_type == CompressionType.ZSTD:
            return self.compressor.compress(data)
        elif self.compression_type == CompressionType.BLOSC:
            return blosc.compress(data, cname='zstd', clevel=self.compression_level)
        else:
            raise ValueError(f"Unsupported compression type: {self.compression_type}")
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data using selected algorithm"""
        if self.compression_type == CompressionType.NONE:
            return compressed_data
        elif self.compression_type == CompressionType.LZ4:
            return lz4.frame.decompress(compressed_data)
        elif self.compression_type == CompressionType.ZSTD:
            return zstd.decompress(compressed_data)
        elif self.compression_type == CompressionType.BLOSC:
            return blosc.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression type: {self.compression_type}")
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum"""
        return hashlib.sha256(data).hexdigest()
    
    def store_data(self, key: str, data: Union[bytes, np.ndarray, pd.DataFrame], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data with compression"""
        start_time = time.time_ns()
        
        try:
            # Serialize data if needed
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
                metadata = metadata or {}
                metadata.update({
                    'dtype': str(data.dtype),
                    'shape': data.shape,
                    'data_type': 'numpy'
                })
            elif isinstance(data, pd.DataFrame):
                data_bytes = data.to_pickle()
                metadata = metadata or {}
                metadata.update({
                    'data_type': 'dataframe',
                    'columns': list(data.columns),
                    'index_name': data.index.name
                })
            elif isinstance(data, bytes):
                data_bytes = data
                metadata = metadata or {}
                metadata.update({'data_type': 'bytes'})
            else:
                data_bytes = pickle.dumps(data)
                metadata = metadata or {}
                metadata.update({'data_type': 'pickle'})
            
            # Compress data
            compressed_data = self._compress_data(data_bytes)
            
            # Calculate checksum if enabled
            checksum = self._calculate_checksum(data_bytes) if self.enable_checksum else ""
            
            # Create data block
            block = DataBlock(
                data=compressed_data,
                metadata=metadata or {},
                compression_type=self.compression_type,
                original_size=len(data_bytes),
                compressed_size=len(compressed_data),
                checksum=checksum
            )
            
            # Store block
            block_id = self._store_block(key, block)
            
            # Update metrics
            end_time = time.time_ns()
            write_latency_us = (end_time - start_time) / 1000
            self._update_write_metrics(len(data_bytes), write_latency_us)
            
            return block_id
            
        except Exception as e:
            logger.error(f"Error storing data for key {key}: {str(e)}")
            raise
    
    def retrieve_data(self, key: str) -> Tuple[Any, Dict[str, Any]]:
        """Retrieve and decompress data"""
        start_time = time.time_ns()
        
        try:
            # Check cache first
            if key in self.cache:
                with self.cache_lock:
                    cached_data, cached_metadata = self.cache[key]
                    return cached_data, cached_metadata
            
            # Retrieve block
            block = self._retrieve_block(key)
            if block is None:
                raise KeyError(f"Data not found for key: {key}")
            
            # Decompress data
            decompressed_data = self._decompress_data(block.data)
            
            # Verify checksum if enabled
            if self.enable_checksum and block.checksum:
                calculated_checksum = self._calculate_checksum(decompressed_data)
                if calculated_checksum != block.checksum:
                    raise ValueError(f"Checksum mismatch for key {key}")
            
            # Deserialize data based on type
            data_type = block.metadata.get('data_type', 'bytes')
            if data_type == 'numpy':
                dtype = np.dtype(block.metadata['dtype'])
                shape = block.metadata['shape']
                data = np.frombuffer(decompressed_data, dtype=dtype).reshape(shape)
            elif data_type == 'dataframe':
                data = pd.read_pickle(decompressed_data)
            elif data_type == 'bytes':
                data = decompressed_data
            else:
                data = pickle.loads(decompressed_data)
            
            # Update cache
            self._update_cache(key, data, block.metadata)
            
            # Update metrics
            end_time = time.time_ns()
            read_latency_us = (end_time - start_time) / 1000
            self._update_read_metrics(len(decompressed_data), read_latency_us)
            
            return data, block.metadata
            
        except Exception as e:
            logger.error(f"Error retrieving data for key {key}: {str(e)}")
            raise
    
    def _store_block(self, key: str, block: DataBlock) -> str:
        """Store compressed block to disk"""
        # Generate unique block ID
        block_id = f"{key}_{int(time.time_ns())}"
        block_path = self.storage_path / f"{block_id}.block"
        
        # Write block to file
        with open(block_path, 'wb') as f:
            # Write header
            header = {
                'compression_type': block.compression_type.value,
                'original_size': block.original_size,
                'compressed_size': block.compressed_size,
                'checksum': block.checksum,
                'metadata': block.metadata,
                'timestamp': block.timestamp
            }
            header_bytes = json.dumps(header).encode('utf-8')
            header_size = len(header_bytes)
            
            f.write(struct.pack('<I', header_size))
            f.write(header_bytes)
            f.write(block.data)
        
        # Update index
        if self.enable_indexing:
            with self.index_lock:
                self.block_index[key] = {
                    'block_id': block_id,
                    'path': str(block_path),
                    'size': block.compressed_size,
                    'timestamp': block.timestamp
                }
        
        return block_id
    
    def _retrieve_block(self, key: str) -> Optional[DataBlock]:
        """Retrieve compressed block from disk"""
        if not self.enable_indexing or key not in self.block_index:
            return None
        
        with self.index_lock:
            block_info = self.block_index[key]
            block_path = Path(block_info['path'])
        
        if not block_path.exists():
            return None
        
        try:
            with open(block_path, 'rb') as f:
                # Read header
                header_size = struct.unpack('<I', f.read(4))[0]
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Read compressed data
                compressed_data = f.read(header['compressed_size'])
                
                # Create block
                return DataBlock(
                    data=compressed_data,
                    metadata=header['metadata'],
                    compression_type=CompressionType(header['compression_type']),
                    original_size=header['original_size'],
                    compressed_size=header['compressed_size'],
                    checksum=header['checksum'],
                    timestamp=header['timestamp']
                )
        
        except Exception as e:
            logger.error(f"Error reading block for key {key}: {str(e)}")
            return None
    
    def _update_cache(self, key: str, data: Any, metadata: Dict[str, Any]):
        """Update data cache"""
        with self.cache_lock:
            # Simple LRU eviction
            if len(self.cache) >= 100:  # Max 100 items
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = (data, metadata)
    
    def _load_index(self):
        """Load block index from disk"""
        if not self.enable_indexing:
            return
        
        index_path = self.storage_path / 'index.json'
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.block_index = json.load(f)
                logger.info(f"Loaded index with {len(self.block_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load index: {str(e)}")
                self.block_index = {}
    
    def _save_index(self):
        """Save block index to disk"""
        if not self.enable_indexing:
            return
        
        index_path = self.storage_path / 'index.json'
        try:
            with self.index_lock:
                with open(index_path, 'w') as f:
                    json.dump(self.block_index, f, indent=2)
            logger.info(f"Saved index with {len(self.block_index)} entries")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def _update_write_metrics(self, bytes_written: int, latency_us: float):
        """Update write metrics"""
        with self.metrics_lock:
            self.metrics.total_writes += 1
            self.metrics.bytes_written += bytes_written
            
            # Update average latency
            if self.metrics.total_writes == 1:
                self.metrics.avg_write_latency_us = latency_us
            else:
                self.metrics.avg_write_latency_us = (
                    (self.metrics.avg_write_latency_us * (self.metrics.total_writes - 1) + latency_us) / 
                    self.metrics.total_writes
                )
    
    def _update_read_metrics(self, bytes_read: int, latency_us: float):
        """Update read metrics"""
        with self.metrics_lock:
            self.metrics.total_reads += 1
            self.metrics.bytes_read += bytes_read
            
            # Update average latency
            if self.metrics.total_reads == 1:
                self.metrics.avg_read_latency_us = latency_us
            else:
                self.metrics.avg_read_latency_us = (
                    (self.metrics.avg_read_latency_us * (self.metrics.total_reads - 1) + latency_us) / 
                    self.metrics.total_reads
                )
    
    def flush_buffer(self):
        """Flush write buffer"""
        with self.buffer_lock:
            # Implementation for buffered writes
            pass
    
    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics"""
        with self.metrics_lock:
            # Calculate compression ratio
            if self.metrics.bytes_written > 0:
                # This is a simplified calculation
                total_compressed = sum(info.get('size', 0) for info in self.block_index.values())
                self.metrics.compression_ratio = total_compressed / self.metrics.bytes_written
            
            # Calculate storage utilization
            total_size = sum(os.path.getsize(self.storage_path / f) 
                           for f in os.listdir(self.storage_path) 
                           if os.path.isfile(self.storage_path / f))
            self.metrics.storage_utilization_mb = total_size / (1024 * 1024)
            
            return StorageMetrics(
                total_writes=self.metrics.total_writes,
                total_reads=self.metrics.total_reads,
                bytes_written=self.metrics.bytes_written,
                bytes_read=self.metrics.bytes_read,
                avg_write_latency_us=self.metrics.avg_write_latency_us,
                avg_read_latency_us=self.metrics.avg_read_latency_us,
                compression_ratio=self.metrics.compression_ratio,
                storage_utilization_mb=self.metrics.storage_utilization_mb,
                timestamp=time.time()
            )
    
    def list_keys(self) -> List[str]:
        """List all stored keys"""
        if not self.enable_indexing:
            return []
        
        with self.index_lock:
            return list(self.block_index.keys())
    
    def delete_key(self, key: str) -> bool:
        """Delete data for key"""
        if not self.enable_indexing or key not in self.block_index:
            return False
        
        with self.index_lock:
            block_info = self.block_index[key]
            block_path = Path(block_info['path'])
            
            # Remove file
            if block_path.exists():
                block_path.unlink()
            
            # Remove from index
            del self.block_index[key]
            
            # Remove from cache
            with self.cache_lock:
                if key in self.cache:
                    del self.cache[key]
        
        return True
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """Cleanup old data blocks"""
        if not self.enable_indexing:
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        keys_to_delete = []
        
        with self.index_lock:
            for key, block_info in self.block_index.items():
                if block_info['timestamp'] < cutoff_time:
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.delete_key(key)
        
        logger.info(f"Cleaned up {len(keys_to_delete)} old data blocks")

class OptimizedFileStorage:
    """Optimized file storage for different data formats"""
    
    def __init__(self, storage_path: str, enable_compression: bool = True):
        self.storage_path = Path(storage_path)
        self.enable_compression = enable_compression
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.metrics = StorageMetrics()
        self.metrics_lock = threading.Lock()
    
    def store_dataframe_parquet(self, df: pd.DataFrame, filename: str, compression: str = 'snappy') -> str:
        """Store DataFrame as Parquet with compression"""
        start_time = time.time_ns()
        
        try:
            file_path = self.storage_path / f"{filename}.parquet"
            
            # Convert to Arrow table and write
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression=compression)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_write_metrics(file_size, latency_us)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing DataFrame as Parquet: {str(e)}")
            raise
    
    def load_dataframe_parquet(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from Parquet file"""
        start_time = time.time_ns()
        
        try:
            file_path = self.storage_path / f"{filename}.parquet"
            
            # Read Parquet file
            df = pd.read_parquet(file_path)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_read_metrics(file_size, latency_us)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from Parquet: {str(e)}")
            raise
    
    def store_dataframe_feather(self, df: pd.DataFrame, filename: str, compression: str = 'zstd') -> str:
        """Store DataFrame as Feather with compression"""
        start_time = time.time_ns()
        
        try:
            file_path = self.storage_path / f"{filename}.feather"
            
            # Write Feather file
            df.to_feather(file_path, compression=compression)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_write_metrics(file_size, latency_us)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing DataFrame as Feather: {str(e)}")
            raise
    
    def load_dataframe_feather(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from Feather file"""
        start_time = time.time_ns()
        
        try:
            file_path = self.storage_path / f"{filename}.feather"
            
            # Read Feather file
            df = pd.read_feather(file_path)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_read_metrics(file_size, latency_us)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from Feather: {str(e)}")
            raise
    
    def store_array_numpy(self, arr: np.ndarray, filename: str, compressed: bool = True) -> str:
        """Store numpy array with optional compression"""
        start_time = time.time_ns()
        
        try:
            if compressed:
                file_path = self.storage_path / f"{filename}.npz"
                np.savez_compressed(file_path, data=arr)
            else:
                file_path = self.storage_path / f"{filename}.npy"
                np.save(file_path, arr)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_write_metrics(file_size, latency_us)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing numpy array: {str(e)}")
            raise
    
    def load_array_numpy(self, filename: str, compressed: bool = True) -> np.ndarray:
        """Load numpy array"""
        start_time = time.time_ns()
        
        try:
            if compressed:
                file_path = self.storage_path / f"{filename}.npz"
                with np.load(file_path) as data:
                    arr = data['data']
            else:
                file_path = self.storage_path / f"{filename}.npy"
                arr = np.load(file_path)
            
            # Update metrics
            end_time = time.time_ns()
            file_size = file_path.stat().st_size
            latency_us = (end_time - start_time) / 1000
            self._update_read_metrics(file_size, latency_us)
            
            return arr
            
        except Exception as e:
            logger.error(f"Error loading numpy array: {str(e)}")
            raise
    
    def _update_write_metrics(self, bytes_written: int, latency_us: float):
        """Update write metrics"""
        with self.metrics_lock:
            self.metrics.total_writes += 1
            self.metrics.bytes_written += bytes_written
            
            if self.metrics.total_writes == 1:
                self.metrics.avg_write_latency_us = latency_us
            else:
                self.metrics.avg_write_latency_us = (
                    (self.metrics.avg_write_latency_us * (self.metrics.total_writes - 1) + latency_us) / 
                    self.metrics.total_writes
                )
    
    def _update_read_metrics(self, bytes_read: int, latency_us: float):
        """Update read metrics"""
        with self.metrics_lock:
            self.metrics.total_reads += 1
            self.metrics.bytes_read += bytes_read
            
            if self.metrics.total_reads == 1:
                self.metrics.avg_read_latency_us = latency_us
            else:
                self.metrics.avg_read_latency_us = (
                    (self.metrics.avg_read_latency_us * (self.metrics.total_reads - 1) + latency_us) / 
                    self.metrics.total_reads
                )
    
    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics"""
        with self.metrics_lock:
            return StorageMetrics(
                total_writes=self.metrics.total_writes,
                total_reads=self.metrics.total_reads,
                bytes_written=self.metrics.bytes_written,
                bytes_read=self.metrics.bytes_read,
                avg_write_latency_us=self.metrics.avg_write_latency_us,
                avg_read_latency_us=self.metrics.avg_read_latency_us,
                compression_ratio=self.metrics.compression_ratio,
                storage_utilization_mb=self.metrics.storage_utilization_mb,
                timestamp=time.time()
            )

# Utility functions
def create_compressed_store(storage_path: str, 
                          compression_type: CompressionType = CompressionType.ZSTD) -> CompressedDataStore:
    """Create compressed data store with default settings"""
    return CompressedDataStore(
        storage_path=storage_path,
        compression_type=compression_type,
        compression_level=3,
        enable_checksum=True,
        enable_indexing=True
    )

def create_file_storage(storage_path: str) -> OptimizedFileStorage:
    """Create optimized file storage"""
    return OptimizedFileStorage(storage_path=storage_path)

def benchmark_compression_algorithms(data: np.ndarray, algorithms: List[CompressionType] = None) -> Dict[str, Dict[str, float]]:
    """Benchmark compression algorithms"""
    if algorithms is None:
        algorithms = [CompressionType.NONE, CompressionType.LZ4, CompressionType.ZSTD, CompressionType.BLOSC]
    
    # Convert to bytes
    data_bytes = data.tobytes()
    original_size = len(data_bytes)
    
    results = {}
    
    for algorithm in algorithms:
        try:
            # Create temporary store
            temp_store = CompressedDataStore(
                storage_path="/tmp/benchmark",
                compression_type=algorithm,
                compression_level=3
            )
            
            # Compression benchmark
            start_time = time.time_ns()
            compressed_data = temp_store._compress_data(data_bytes)
            compress_time = (time.time_ns() - start_time) / 1000  # microseconds
            
            # Decompression benchmark
            start_time = time.time_ns()
            decompressed_data = temp_store._decompress_data(compressed_data)
            decompress_time = (time.time_ns() - start_time) / 1000  # microseconds
            
            # Calculate metrics
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size
            
            results[algorithm.value] = {
                'compression_ratio': compression_ratio,
                'compress_time_us': compress_time,
                'decompress_time_us': decompress_time,
                'total_time_us': compress_time + decompress_time,
                'compressed_size': compressed_size,
                'original_size': original_size
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking {algorithm.value}: {str(e)}")
            results[algorithm.value] = {'error': str(e)}
    
    return results

def optimize_storage_format(df: pd.DataFrame, storage_path: str) -> Dict[str, Dict[str, Any]]:
    """Test different storage formats and return performance metrics"""
    storage = OptimizedFileStorage(storage_path)
    results = {}
    
    # Test Parquet
    try:
        start_time = time.time_ns()
        storage.store_dataframe_parquet(df, "test_parquet")
        write_time = (time.time_ns() - start_time) / 1000
        
        start_time = time.time_ns()
        loaded_df = storage.load_dataframe_parquet("test_parquet")
        read_time = (time.time_ns() - start_time) / 1000
        
        file_size = (storage.storage_path / "test_parquet.parquet").stat().st_size
        
        results['parquet'] = {
            'write_time_us': write_time,
            'read_time_us': read_time,
            'file_size': file_size,
            'compression_ratio': file_size / df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        results['parquet'] = {'error': str(e)}
    
    # Test Feather
    try:
        start_time = time.time_ns()
        storage.store_dataframe_feather(df, "test_feather")
        write_time = (time.time_ns() - start_time) / 1000
        
        start_time = time.time_ns()
        loaded_df = storage.load_dataframe_feather("test_feather")
        read_time = (time.time_ns() - start_time) / 1000
        
        file_size = (storage.storage_path / "test_feather.feather").stat().st_size
        
        results['feather'] = {
            'write_time_us': write_time,
            'read_time_us': read_time,
            'file_size': file_size,
            'compression_ratio': file_size / df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        results['feather'] = {'error': str(e)}
    
    return results
