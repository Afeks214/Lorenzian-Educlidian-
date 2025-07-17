"""
Scalable Data Loader for massive CSV files with memory optimization
"""

import os
import gc
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import mmap
import csv
import io
import gzip
import bz2
import lzma
import pickle
import joblib
from functools import partial
import threading
from queue import Queue, Empty
import weakref

from .config import DataPipelineConfig
from .exceptions import DataLoadingException, PerformanceException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataChunk:
    """Represents a chunk of data with metadata"""
    data: pd.DataFrame
    chunk_id: int
    start_row: int
    end_row: int
    file_path: str
    timestamp: float
    memory_usage: float
    
    def __post_init__(self):
        """Calculate memory usage if not provided"""
        if self.memory_usage == 0:
            self.memory_usage = self.data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

@dataclass
class FileMetadata:
    """Metadata for data files"""
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    compression: Optional[str]
    encoding: str
    delimiter: str
    last_modified: float
    
class ScalableDataLoader:
    """
    High-performance data loader for massive CSV files with memory optimization
    and parallel processing capabilities.
    """
    
    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()
        self.file_cache: Dict[str, FileMetadata] = {}
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        self.performance_stats = PerformanceStats()
        self._lock = threading.Lock()
        
        # Setup cleanup for cache
        self._cleanup_cache_on_exit()
    
    def _cleanup_cache_on_exit(self):
        """Setup cleanup for cache on exit"""
        import atexit
        atexit.register(self._cleanup_cache)
    
    def _cleanup_cache(self):
        """Cleanup cache and temporary files"""
        logger.info("Cleaning up data loader cache")
        self.file_cache.clear()
        gc.collect()
    
    def get_file_metadata(self, file_path: str, force_refresh: bool = False) -> FileMetadata:
        """
        Get metadata for a file, with caching for performance
        """
        file_path = str(Path(file_path).resolve())
        
        # Check cache first
        if not force_refresh and file_path in self.file_cache:
            cached_meta = self.file_cache[file_path]
            file_stat = os.stat(file_path)
            if cached_meta.last_modified == file_stat.st_mtime:
                return cached_meta
        
        # Generate new metadata
        try:
            metadata = self._generate_file_metadata(file_path)
            self.file_cache[file_path] = metadata
            return metadata
        except Exception as e:
            raise DataLoadingException(f"Failed to generate metadata for {file_path}: {str(e)}")
    
    def _generate_file_metadata(self, file_path: str) -> FileMetadata:
        """Generate metadata for a file"""
        file_stat = os.stat(file_path)
        file_size_bytes = file_stat.st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Detect compression
        compression = self._detect_compression(file_path)
        
        # Sample file to get structure
        sample_df = self._sample_file(file_path, sample_size=1000)
        
        # Detect encoding and delimiter
        encoding = self._detect_encoding(file_path)
        delimiter = self._detect_delimiter(file_path)
        
        # Estimate row count
        row_count = self._estimate_row_count(file_path, sample_df)
        
        return FileMetadata(
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            row_count=row_count,
            column_count=len(sample_df.columns),
            columns=sample_df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            compression=compression,
            encoding=encoding,
            delimiter=delimiter,
            last_modified=file_stat.st_mtime
        )
    
    def _detect_compression(self, file_path: str) -> Optional[str]:
        """Detect compression type from file extension"""
        suffix = Path(file_path).suffix.lower()
        compression_map = {
            '.gz': 'gzip',
            '.bz2': 'bz2',
            '.xz': 'xz',
            '.lzma': 'lzma'
        }
        return compression_map.get(suffix)
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        import chardet
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _detect_delimiter(self, file_path: str) -> str:
        """Detect CSV delimiter"""
        try:
            with open(file_path, 'r', encoding=self._detect_encoding(file_path)) as f:
                first_line = f.readline()
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(first_line).delimiter
                return delimiter
        except Exception:
            return ','
    
    def _sample_file(self, file_path: str, sample_size: int = 1000) -> pd.DataFrame:
        """Sample file to understand structure"""
        try:
            compression = self._detect_compression(file_path)
            encoding = self._detect_encoding(file_path)
            delimiter = self._detect_delimiter(file_path)
            
            sample_df = pd.read_csv(
                file_path,
                nrows=sample_size,
                compression=compression,
                encoding=encoding,
                delimiter=delimiter,
                low_memory=False
            )
            return sample_df
        except Exception as e:
            raise DataLoadingException(f"Failed to sample file {file_path}: {str(e)}")
    
    def _estimate_row_count(self, file_path: str, sample_df: pd.DataFrame) -> int:
        """Estimate total row count in file"""
        try:
            file_size = os.path.getsize(file_path)
            if len(sample_df) == 0:
                return 0
            
            # Estimate based on sample
            sample_size_bytes = len(sample_df.to_csv(index=False).encode('utf-8'))
            estimated_rows = int((file_size / sample_size_bytes) * len(sample_df))
            return max(estimated_rows, len(sample_df))
        except Exception:
            return len(sample_df)
    
    def load_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load entire file into memory (use with caution for large files)
        """
        metadata = self.get_file_metadata(file_path)
        
        # Check memory requirements
        estimated_memory = metadata.file_size_mb * 3  # Conservative estimate
        if estimated_memory > self.config.memory_limit_mb:
            raise PerformanceException(
                f"File {file_path} estimated memory usage ({estimated_memory:.2f}MB) "
                f"exceeds limit ({self.config.memory_limit_mb}MB). Use load_chunks() instead."
            )
        
        try:
            df = pd.read_csv(
                file_path,
                compression=metadata.compression,
                encoding=metadata.encoding,
                delimiter=metadata.delimiter,
                **kwargs
            )
            
            # Update performance stats
            self.performance_stats.files_loaded += 1
            self.performance_stats.total_rows += len(df)
            
            return df
        except Exception as e:
            raise DataLoadingException(f"Failed to load file {file_path}: {str(e)}")
    
    def load_chunks(self, file_path: str, chunk_size: Optional[int] = None, **kwargs) -> Iterator[DataChunk]:
        """
        Load file in chunks for memory-efficient processing
        """
        metadata = self.get_file_metadata(file_path)
        actual_chunk_size = chunk_size or self.config.get_optimal_chunk_size(metadata.file_size_mb)
        
        try:
            chunk_reader = pd.read_csv(
                file_path,
                chunksize=actual_chunk_size,
                compression=metadata.compression,
                encoding=metadata.encoding,
                delimiter=metadata.delimiter,
                **kwargs
            )
            
            chunk_id = 0
            current_row = 0
            
            for chunk_df in chunk_reader:
                # Memory check
                if self.memory_monitor.check_memory_usage():
                    logger.warning("Memory usage high, triggering garbage collection")
                    gc.collect()
                
                chunk = DataChunk(
                    data=chunk_df,
                    chunk_id=chunk_id,
                    start_row=current_row,
                    end_row=current_row + len(chunk_df),
                    file_path=file_path,
                    timestamp=time.time(),
                    memory_usage=0  # Will be calculated in __post_init__
                )
                
                # Update performance stats
                self.performance_stats.chunks_processed += 1
                self.performance_stats.total_rows += len(chunk_df)
                
                yield chunk
                
                chunk_id += 1
                current_row += len(chunk_df)
        
        except Exception as e:
            raise DataLoadingException(f"Failed to load chunks from {file_path}: {str(e)}")
    
    def load_multiple_files(self, file_paths: List[str], parallel: bool = True, **kwargs) -> Iterator[DataChunk]:
        """
        Load multiple files in parallel or sequentially
        """
        if parallel and self.config.enable_parallel_processing:
            yield from self._load_multiple_files_parallel(file_paths, **kwargs)
        else:
            yield from self._load_multiple_files_sequential(file_paths, **kwargs)
    
    def _load_multiple_files_sequential(self, file_paths: List[str], **kwargs) -> Iterator[DataChunk]:
        """Load multiple files sequentially"""
        for file_path in file_paths:
            yield from self.load_chunks(file_path, **kwargs)
    
    def _load_multiple_files_parallel(self, file_paths: List[str], **kwargs) -> Iterator[DataChunk]:
        """Load multiple files in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._load_file_chunks, file_path, **kwargs): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        yield chunk
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    raise DataLoadingException(f"Failed to load file {file_path}: {str(e)}")
    
    def _load_file_chunks(self, file_path: str, **kwargs) -> List[DataChunk]:
        """Load all chunks from a single file"""
        return list(self.load_chunks(file_path, **kwargs))
    
    def load_time_range(self, file_path: str, start_time: str, end_time: str, 
                       time_column: str = 'timestamp', **kwargs) -> Iterator[DataChunk]:
        """
        Load data within a specific time range
        """
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        for chunk in self.load_chunks(file_path, **kwargs):
            # Convert time column to datetime if needed
            if time_column in chunk.data.columns:
                chunk.data[time_column] = pd.to_datetime(chunk.data[time_column])
                
                # Filter by time range
                mask = (chunk.data[time_column] >= start_time) & (chunk.data[time_column] <= end_time)
                filtered_data = chunk.data[mask]
                
                if not filtered_data.empty:
                    filtered_chunk = DataChunk(
                        data=filtered_data,
                        chunk_id=chunk.chunk_id,
                        start_row=chunk.start_row,
                        end_row=chunk.end_row,
                        file_path=chunk.file_path,
                        timestamp=chunk.timestamp,
                        memory_usage=0  # Will be calculated
                    )
                    yield filtered_chunk
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'files_loaded': self.performance_stats.files_loaded,
            'chunks_processed': self.performance_stats.chunks_processed,
            'total_rows': self.performance_stats.total_rows,
            'memory_usage_mb': self.memory_monitor.get_memory_usage(),
            'peak_memory_mb': self.memory_monitor.peak_memory_usage,
            'cache_size': len(self.file_cache)
        }
    
    def clear_cache(self):
        """Clear file metadata cache"""
        with self._lock:
            self.file_cache.clear()
            logger.info("File metadata cache cleared")


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed"""
    
    def __init__(self, memory_limit_mb: int):
        self.memory_limit_mb = memory_limit_mb
        self.peak_memory_usage = 0
        self.check_count = 0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
        return memory_mb
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold"""
        self.check_count += 1
        current_usage = self.get_memory_usage()
        
        if current_usage > self.memory_limit_mb * 0.8:  # 80% threshold
            return True
        return False
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        logger.info(f"Memory cleanup triggered. Usage: {self.get_memory_usage():.2f}MB")


class PerformanceStats:
    """Track performance statistics"""
    
    def __init__(self):
        self.files_loaded = 0
        self.chunks_processed = 0
        self.total_rows = 0
        self.start_time = time.time()
    
    def get_throughput(self) -> float:
        """Get processing throughput in rows per second"""
        elapsed = time.time() - self.start_time
        return self.total_rows / elapsed if elapsed > 0 else 0
    
    def reset(self):
        """Reset statistics"""
        self.files_loaded = 0
        self.chunks_processed = 0
        self.total_rows = 0
        self.start_time = time.time()


# Context manager for memory-efficient processing
@contextmanager
def memory_efficient_processing(memory_limit_mb: int = 1024):
    """Context manager for memory-efficient data processing"""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        # Cleanup
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_diff = final_memory - initial_memory
        
        if memory_diff > memory_limit_mb * 0.5:  # If memory increased significantly
            logger.warning(f"Memory usage increased by {memory_diff:.2f}MB during processing")