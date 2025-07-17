"""
Memory-mapped test data and optimization for faster I/O operations.
Agent 4 Mission: Test Data Management & Caching System
"""
import os
import mmap
import struct
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import gzip
import ast

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import asyncio
from contextlib import contextmanager
import pytest
from unittest.mock import Mock

@dataclass
class MemoryMapConfig:
    """Configuration for memory-mapped data."""
    chunk_size: int = 8192  # 8KB chunks
    compression_level: int = 3
    compression_type: str = "lz4"  # lz4, zstd, gzip
    max_memory_mb: int = 1024  # 1GB max memory usage
    preload_data: bool = True
    use_async_io: bool = True
    parallel_workers: int = 4

class MemoryMappedTestData:
    """Memory-mapped test data for high-performance I/O."""
    
    def __init__(self, config: MemoryMapConfig):
        self.config = config
        self.active_mappings = {}
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.cache_dir = Path(".pytest_cache/mmap_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Preloaded data cache
        self.preloaded_data = {}
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.parallel_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.parallel_workers)
    
    def create_memory_mapped_file(self, data: Union[pd.DataFrame, np.ndarray, Dict], 
                                 filename: str) -> str:
        """Create memory-mapped file from data."""
        file_path = self.cache_dir / f"{filename}.mmap"
        
        with self.lock:
            # Serialize data
            if isinstance(data, pd.DataFrame):
                serialized_data = self._serialize_dataframe(data)
            elif isinstance(data, np.ndarray):
                serialized_data = self._serialize_numpy_array(data)
            elif isinstance(data, dict):
                serialized_data = self._serialize_dict(data)
            else:
                serialized_data = pickle.dumps(data)
            
            # Compress data
            compressed_data = self._compress_data(serialized_data)
            
            # Write to file
            with open(file_path, 'wb') as f:
                # Write header with metadata
                header = struct.pack('I', len(compressed_data))
                f.write(header)
                f.write(compressed_data)
            
            # Update memory usage tracking
            self.memory_usage += len(compressed_data)
            
            return str(file_path)
    
    def _serialize_dataframe(self, df: pd.DataFrame) -> bytes:
        """Serialize DataFrame efficiently."""
        # Use optimized DataFrame serialization
        buffer = df.to_parquet(None, engine='pyarrow', compression=None)
        return buffer
    
    def _serialize_numpy_array(self, arr: np.ndarray) -> bytes:
        """Serialize numpy array efficiently."""
        # Use numpy's native serialization
        buffer = arr.tobytes()
        dtype_str = str(arr.dtype)
        shape_str = str(arr.shape)
        
        # Create header with metadata
        header = f"{dtype_str}|{shape_str}".encode('utf-8')
        header_size = struct.pack('I', len(header))
        
        return header_size + header + buffer
    
    def _serialize_dict(self, data: Dict) -> bytes:
        """Serialize dictionary efficiently."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using configured compression."""
        if self.config.compression_type == "lz4" and LZ4_AVAILABLE:
            return lz4.frame.compress(data, compression_level=self.config.compression_level)
        elif self.config.compression_type == "zstd" and ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=self.config.compression_level)
            return cctx.compress(data)
        elif self.config.compression_type == "gzip":
            return gzip.compress(data, compresslevel=self.config.compression_level)
        else:
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using configured compression."""
        if self.config.compression_type == "lz4" and LZ4_AVAILABLE:
            return lz4.frame.decompress(data)
        elif self.config.compression_type == "zstd" and ZSTD_AVAILABLE:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        elif self.config.compression_type == "gzip":
            return gzip.decompress(data)
        else:
            return data
    
    @contextmanager
    def memory_mapped_file(self, filename: str):
        """Context manager for memory-mapped file access."""
        file_path = self.cache_dir / f"{filename}.mmap"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Memory-mapped file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                try:
                    yield mm
                finally:
                    pass
    
    def load_memory_mapped_data(self, filename: str) -> Any:
        """Load data from memory-mapped file."""
        if self.config.preload_data and filename in self.preloaded_data:
            return self.preloaded_data[filename]
        
        with self.memory_mapped_file(filename) as mm:
            # Read header
            header_size = struct.unpack('I', mm[:4])[0]
            compressed_data = mm[4:4+header_size]
            
            # Decompress data
            serialized_data = self._decompress_data(compressed_data)
            
            # Deserialize data
            data = self._deserialize_data(serialized_data)
            
            # Cache if preloading is enabled
            if self.config.preload_data:
                self.preloaded_data[filename] = data
            
            return data
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data based on type."""
        try:
            # Try DataFrame first (parquet format)
            import io
            return pd.read_parquet(io.BytesIO(data))
        except (pd.errors.ParserError, ValueError, ImportError) as e:
            try:
                # Try numpy array
                header_size = struct.unpack('I', data[:4])[0]
                header = data[4:4+header_size].decode('utf-8')
                dtype_str, shape_str = header.split('|')
                
                dtype = np.dtype(dtype_str)
                shape = ast.literal_eval(shape_str)
                
                array_data = data[4+header_size:]
                arr = np.frombuffer(array_data, dtype=dtype)
                return arr.reshape(shape)
            except (struct.error, ValueError, TypeError, AttributeError) as e:
                # Fall back to pickle
                try:
                    return pickle.loads(data)
                except (pickle.UnpicklingError, EOFError, ValueError) as e:
                    raise ValueError(f"Could not deserialize data: {e}")
    
    async def async_load_data(self, filename: str) -> Any:
        """Asynchronously load data from memory-mapped file."""
        if self.config.preload_data and filename in self.preloaded_data:
            return self.preloaded_data[filename]
        
        file_path = self.cache_dir / f"{filename}.mmap"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Memory-mapped file not found: {file_path}")
        
        async with aiofiles.open(file_path, 'rb') as f:
            # Read header
            header_bytes = await f.read(4)
            header_size = struct.unpack('I', header_bytes)[0]
            
            # Read compressed data
            compressed_data = await f.read(header_size)
            
            # Decompress and deserialize in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.thread_pool,
                self._decompress_and_deserialize,
                compressed_data
            )
            
            # Cache if preloading is enabled
            if self.config.preload_data:
                self.preloaded_data[filename] = data
            
            return data
    
    def _decompress_and_deserialize(self, compressed_data: bytes) -> Any:
        """Helper method for async decompression and deserialization."""
        serialized_data = self._decompress_data(compressed_data)
        return self._deserialize_data(serialized_data)
    
    def preload_data_files(self, filenames: List[str]) -> None:
        """Preload multiple data files in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = [
                executor.submit(self.load_memory_mapped_data, filename)
                for filename in filenames
            ]
            
            for future in futures:
                try:
                    future.result()  # Wait for completion
                except (FileNotFoundError, OSError, IOError) as e:
                    print(f"Error preloading data file: {e}")
                except (ValueError, TypeError, pickle.UnpicklingError) as e:
                    print(f"Error deserializing data: {e}")
                except Exception as e:
                    print(f"Unexpected error preloading data: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return {
            "total_memory_mb": self.memory_usage / (1024 * 1024),
            "max_memory_mb": self.config.max_memory_mb,
            "memory_utilization": self.memory_usage / (self.config.max_memory_mb * 1024 * 1024),
            "active_mappings": len(self.active_mappings),
            "preloaded_items": len(self.preloaded_data)
        }
    
    def cleanup_memory_mappings(self) -> None:
        """Clean up memory mappings to free memory."""
        with self.lock:
            self.active_mappings.clear()
            self.preloaded_data.clear()
            self.memory_usage = 0
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by cleaning up unused mappings."""
        if self.memory_usage > self.config.max_memory_mb * 1024 * 1024:
            # Clear least recently used data
            self.cleanup_memory_mappings()
    
    def close(self) -> None:
        """Close thread pools and clean up resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.cleanup_memory_mappings()

class CompressionManager:
    """Manages different compression strategies for test data."""
    
    def __init__(self):
        self.compression_stats = {}
    
    def benchmark_compression(self, data: bytes, algorithms: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark different compression algorithms."""
        if algorithms is None:
            algorithms = ["lz4", "zstd", "gzip"]
        
        results = {}
        
        for algorithm in algorithms:
            start_time = datetime.now()
            
            if algorithm == "lz4":
                compressed = lz4.frame.compress(data)
                decompressed = lz4.frame.decompress(compressed)
            elif algorithm == "zstd":
                cctx = zstd.ZstdCompressor(level=3)
                compressed = cctx.compress(data)
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(compressed)
            elif algorithm == "gzip":
                compressed = gzip.compress(data, compresslevel=6)
                decompressed = gzip.decompress(compressed)
            
            end_time = datetime.now()
            
            results[algorithm] = {
                "original_size": len(data),
                "compressed_size": len(compressed),
                "compression_ratio": len(compressed) / len(data),
                "compression_time_ms": (end_time - start_time).total_seconds() * 1000,
                "is_valid": data == decompressed
            }
        
        return results
    
    def select_optimal_compression(self, data: bytes, 
                                 priority: str = "speed") -> Tuple[str, Dict[str, float]]:
        """Select optimal compression algorithm based on priority."""
        benchmark_results = self.benchmark_compression(data)
        
        if priority == "speed":
            # Select fastest algorithm
            fastest = min(benchmark_results.items(), 
                         key=lambda x: x[1]["compression_time_ms"])
            return fastest[0], fastest[1]
        
        elif priority == "size":
            # Select best compression ratio
            best_ratio = min(benchmark_results.items(), 
                           key=lambda x: x[1]["compression_ratio"])
            return best_ratio[0], best_ratio[1]
        
        elif priority == "balanced":
            # Select balanced approach
            scores = {}
            for alg, stats in benchmark_results.items():
                # Normalize metrics (lower is better)
                time_score = stats["compression_time_ms"] / 1000  # Convert to seconds
                size_score = stats["compression_ratio"]
                
                # Balanced score (equal weight)
                scores[alg] = (time_score + size_score) / 2
            
            best_balanced = min(scores.items(), key=lambda x: x[1])
            return best_balanced[0], benchmark_results[best_balanced[0]]
        
        else:
            raise ValueError(f"Unknown priority: {priority}")

class TestDataPreloader:
    """Preloads and manages test data for faster access."""
    
    def __init__(self, memory_manager: MemoryMappedTestData):
        self.memory_manager = memory_manager
        self.preload_queue = []
        self.loading_status = {}
        self.lock = threading.Lock()
    
    def add_to_preload_queue(self, filename: str, priority: int = 0) -> None:
        """Add file to preload queue with priority."""
        with self.lock:
            self.preload_queue.append((priority, filename))
            self.preload_queue.sort(key=lambda x: x[0], reverse=True)
    
    def preload_next_batch(self, batch_size: int = 10) -> List[str]:
        """Preload next batch of files."""
        with self.lock:
            batch = self.preload_queue[:batch_size]
            self.preload_queue = self.preload_queue[batch_size:]
            
            filenames = [item[1] for item in batch]
        
        # Load files in parallel
        successful_loads = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.memory_manager.load_memory_mapped_data, filename): filename
                for filename in filenames
            }
            
            for future in futures:
                filename = futures[future]
                try:
                    future.result()
                    successful_loads.append(filename)
                    self.loading_status[filename] = "loaded"
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    self.loading_status[filename] = "error"
        
        return successful_loads
    
    def get_loading_status(self) -> Dict[str, str]:
        """Get current loading status."""
        with self.lock:
            return self.loading_status.copy()
    
    def clear_preload_queue(self) -> None:
        """Clear preload queue."""
        with self.lock:
            self.preload_queue.clear()
            self.loading_status.clear()

# Global instances
memory_config = MemoryMapConfig()
memory_manager = MemoryMappedTestData(memory_config)
compression_manager = CompressionManager()
preloader = TestDataPreloader(memory_manager)

# Pytest fixtures
@pytest.fixture(scope="session")
def memory_mapped_data():
    """Provide memory-mapped test data manager."""
    yield memory_manager
    memory_manager.close()

@pytest.fixture(scope="session")
def compression_manager_fixture():
    """Provide compression manager."""
    return compression_manager

@pytest.fixture
def test_data_preloader():
    """Provide test data preloader."""
    return preloader

@pytest.fixture
def sample_memory_mapped_data():
    """Create sample memory-mapped data for testing."""
    # Create sample DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
        'price': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create memory-mapped file
    filename = "sample_test_data"
    memory_manager.create_memory_mapped_file(data, filename)
    
    yield filename
    
    # Cleanup
    file_path = memory_manager.cache_dir / f"{filename}.mmap"
    if file_path.exists():
        file_path.unlink()

# Test classes
class TestMemoryMappedTestData:
    """Tests for memory-mapped test data."""
    
    def test_dataframe_serialization(self):
        """Test DataFrame serialization and deserialization."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        
        # Create test DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'price': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Create memory-mapped file
        filename = "test_dataframe"
        manager.create_memory_mapped_file(df, filename)
        
        # Load data back
        loaded_df = manager.load_memory_mapped_data(filename)
        
        # Verify data integrity
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
        
        # Cleanup
        manager.close()
        file_path = manager.cache_dir / f"{filename}.mmap"
        if file_path.exists():
            file_path.unlink()
    
    def test_numpy_array_serialization(self):
        """Test numpy array serialization and deserialization."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        
        # Create test array
        arr = np.random.randn(100, 10)
        
        # Create memory-mapped file
        filename = "test_array"
        manager.create_memory_mapped_file(arr, filename)
        
        # Load data back
        loaded_arr = manager.load_memory_mapped_data(filename)
        
        # Verify data integrity
        assert isinstance(loaded_arr, np.ndarray)
        assert loaded_arr.shape == arr.shape
        assert loaded_arr.dtype == arr.dtype
        np.testing.assert_array_equal(loaded_arr, arr)
        
        # Cleanup
        manager.close()
        file_path = manager.cache_dir / f"{filename}.mmap"
        if file_path.exists():
            file_path.unlink()
    
    def test_dict_serialization(self):
        """Test dictionary serialization and deserialization."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        
        # Create test dictionary
        data = {
            "config": {"param1": 1.0, "param2": "test"},
            "results": [1, 2, 3, 4, 5],
            "metadata": {"created": "2023-01-01"}
        }
        
        # Create memory-mapped file
        filename = "test_dict"
        manager.create_memory_mapped_file(data, filename)
        
        # Load data back
        loaded_data = manager.load_memory_mapped_data(filename)
        
        # Verify data integrity
        assert isinstance(loaded_data, dict)
        assert loaded_data == data
        
        # Cleanup
        manager.close()
        file_path = manager.cache_dir / f"{filename}.mmap"
        if file_path.exists():
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_async_data_loading(self):
        """Test asynchronous data loading."""
        manager = MemoryMappedTestData(MemoryMapConfig(use_async_io=True))
        
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'price': np.random.randn(100)
        })
        
        # Create memory-mapped file
        filename = "test_async"
        manager.create_memory_mapped_file(df, filename)
        
        # Load data asynchronously
        loaded_df = await manager.async_load_data(filename)
        
        # Verify data integrity
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        
        # Cleanup
        manager.close()
        file_path = manager.cache_dir / f"{filename}.mmap"
        if file_path.exists():
            file_path.unlink()
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        
        # Create test data
        df = pd.DataFrame({
            'data': np.random.randn(1000)
        })
        
        # Check initial memory usage
        initial_usage = manager.get_memory_usage()
        
        # Create memory-mapped file
        filename = "test_memory"
        manager.create_memory_mapped_file(df, filename)
        
        # Check memory usage after creation
        after_usage = manager.get_memory_usage()
        
        assert after_usage["total_memory_mb"] > initial_usage["total_memory_mb"]
        
        # Cleanup
        manager.close()
        file_path = manager.cache_dir / f"{filename}.mmap"
        if file_path.exists():
            file_path.unlink()

class TestCompressionManager:
    """Tests for compression manager."""
    
    def test_compression_benchmark(self):
        """Test compression benchmarking."""
        manager = CompressionManager()
        
        # Create test data
        data = b"Hello, World! " * 1000
        
        # Benchmark compression
        results = manager.benchmark_compression(data)
        
        assert "lz4" in results
        assert "zstd" in results
        assert "gzip" in results
        
        for alg, stats in results.items():
            assert "original_size" in stats
            assert "compressed_size" in stats
            assert "compression_ratio" in stats
            assert "compression_time_ms" in stats
            assert stats["is_valid"] == True
    
    def test_optimal_compression_selection(self):
        """Test optimal compression selection."""
        manager = CompressionManager()
        
        # Create test data
        data = b"Test data for compression" * 100
        
        # Test different priorities
        speed_alg, speed_stats = manager.select_optimal_compression(data, "speed")
        size_alg, size_stats = manager.select_optimal_compression(data, "size")
        balanced_alg, balanced_stats = manager.select_optimal_compression(data, "balanced")
        
        assert speed_alg in ["lz4", "zstd", "gzip"]
        assert size_alg in ["lz4", "zstd", "gzip"]
        assert balanced_alg in ["lz4", "zstd", "gzip"]
        
        assert speed_stats["is_valid"] == True
        assert size_stats["is_valid"] == True
        assert balanced_stats["is_valid"] == True

class TestTestDataPreloader:
    """Tests for test data preloader."""
    
    def test_preload_queue_management(self):
        """Test preload queue management."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        preloader = TestDataPreloader(manager)
        
        # Add items to queue
        preloader.add_to_preload_queue("file1.mmap", priority=1)
        preloader.add_to_preload_queue("file2.mmap", priority=3)
        preloader.add_to_preload_queue("file3.mmap", priority=2)
        
        # Check queue ordering (higher priority first)
        assert preloader.preload_queue[0][1] == "file2.mmap"
        assert preloader.preload_queue[1][1] == "file3.mmap"
        assert preloader.preload_queue[2][1] == "file1.mmap"
        
        # Clear queue
        preloader.clear_preload_queue()
        assert len(preloader.preload_queue) == 0
    
    def test_loading_status_tracking(self):
        """Test loading status tracking."""
        manager = MemoryMappedTestData(MemoryMapConfig())
        preloader = TestDataPreloader(manager)
        
        # Initially empty
        status = preloader.get_loading_status()
        assert len(status) == 0
        
        # Add some status
        preloader.loading_status["file1.mmap"] = "loaded"
        preloader.loading_status["file2.mmap"] = "error"
        
        status = preloader.get_loading_status()
        assert status["file1.mmap"] == "loaded"
        assert status["file2.mmap"] == "error"