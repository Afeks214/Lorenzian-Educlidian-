"""
Data streaming implementation with minimal memory footprint
"""

import time
import threading
import queue
import pandas as pd
import numpy as np
from typing import Iterator, Callable, Optional, Any, Dict, List, Union, AsyncIterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import gc
import psutil
from pathlib import Path
import weakref
import asyncio
from contextlib import asynccontextmanager

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataStreamingException
from ..core.data_loader import DataChunk, ScalableDataLoader

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    buffer_size: int = 1000
    max_queue_size: int = 10
    timeout_seconds: float = 30.0
    enable_backpressure: bool = True
    memory_threshold_mb: float = 500.0
    enable_compression: bool = True
    enable_batching: bool = True
    batch_size: int = 100

class DataStreamer:
    """
    High-performance data streamer with minimal memory footprint
    """
    
    def __init__(self, config: Optional[DataPipelineConfig] = None,
                 stream_config: Optional[StreamConfig] = None):
        self.config = config or DataPipelineConfig()
        self.stream_config = stream_config or StreamConfig()
        self.data_loader = ScalableDataLoader(self.config)
        
        # Streaming state
        self._is_streaming = False
        self._stream_queue = queue.Queue(maxsize=self.stream_config.max_queue_size)
        self._error_queue = queue.Queue()
        self._stats = StreamingStats()
        self._lock = threading.Lock()
        
        # Memory monitoring
        self._memory_monitor = MemoryMonitor(self.stream_config.memory_threshold_mb)
        
        # Cleanup on exit
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.stop_streaming()
        logger.info("DataStreamer cleanup completed")
    
    def stream_file(self, file_path: str, 
                   transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                   filter_func: Optional[Callable[[pd.DataFrame], bool]] = None,
                   **kwargs) -> Iterator[DataChunk]:
        """
        Stream data from a file with optional transformation and filtering
        """
        try:
            self._is_streaming = True
            self._stats.reset()
            
            # Stream chunks
            for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                # Check memory usage
                if self._memory_monitor.check_memory_pressure():
                    logger.warning("Memory pressure detected, triggering cleanup")
                    gc.collect()
                
                # Apply transformations
                processed_chunk = self._process_chunk(chunk, transform_func, filter_func)
                
                if processed_chunk is not None:
                    # Update stats
                    self._stats.update(processed_chunk)
                    yield processed_chunk
                
                # Check for streaming stop
                if not self._is_streaming:
                    break
                    
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {str(e)}")
            raise DataStreamingException(f"Failed to stream file {file_path}: {str(e)}")
        finally:
            self._is_streaming = False
    
    def stream_multiple_files(self, file_paths: List[str],
                            transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                            filter_func: Optional[Callable[[pd.DataFrame], bool]] = None,
                            parallel: bool = True,
                            **kwargs) -> Iterator[DataChunk]:
        """
        Stream data from multiple files
        """
        if parallel and self.config.enable_parallel_processing:
            yield from self._stream_multiple_files_parallel(
                file_paths, transform_func, filter_func, **kwargs
            )
        else:
            yield from self._stream_multiple_files_sequential(
                file_paths, transform_func, filter_func, **kwargs
            )
    
    def _stream_multiple_files_sequential(self, file_paths: List[str],
                                        transform_func: Optional[Callable] = None,
                                        filter_func: Optional[Callable] = None,
                                        **kwargs) -> Iterator[DataChunk]:
        """Stream multiple files sequentially"""
        for file_path in file_paths:
            yield from self.stream_file(file_path, transform_func, filter_func, **kwargs)
    
    def _stream_multiple_files_parallel(self, file_paths: List[str],
                                      transform_func: Optional[Callable] = None,
                                      filter_func: Optional[Callable] = None,
                                      **kwargs) -> Iterator[DataChunk]:
        """Stream multiple files in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit streaming tasks
            futures = [
                executor.submit(self._stream_file_to_queue, file_path, transform_func, filter_func, **kwargs)
                for file_path in file_paths
            ]
            
            # Collect results from queue
            active_futures = set(futures)
            while active_futures:
                try:
                    chunk = self._stream_queue.get(timeout=self.stream_config.timeout_seconds)
                    if chunk is None:  # End of stream marker
                        break
                    yield chunk
                    self._stream_queue.task_done()
                except queue.Empty:
                    # Check if all futures are done
                    active_futures = {f for f in active_futures if not f.done()}
                    if not active_futures:
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error in parallel streaming: {str(e)}")
                    break
    
    def _stream_file_to_queue(self, file_path: str,
                            transform_func: Optional[Callable] = None,
                            filter_func: Optional[Callable] = None,
                            **kwargs):
        """Stream file chunks to queue"""
        try:
            for chunk in self.stream_file(file_path, transform_func, filter_func, **kwargs):
                self._stream_queue.put(chunk)
        except Exception as e:
            self._error_queue.put(e)
        finally:
            self._stream_queue.put(None)  # End of stream marker
    
    def _process_chunk(self, chunk: DataChunk,
                      transform_func: Optional[Callable] = None,
                      filter_func: Optional[Callable] = None) -> Optional[DataChunk]:
        """Process a data chunk with transformation and filtering"""
        try:
            data = chunk.data
            
            # Apply filter first (to reduce data size)
            if filter_func and not filter_func(data):
                return None
            
            # Apply transformation
            if transform_func:
                data = transform_func(data)
                if data is None or data.empty:
                    return None
            
            # Create new chunk with processed data
            processed_chunk = DataChunk(
                data=data,
                chunk_id=chunk.chunk_id,
                start_row=chunk.start_row,
                end_row=chunk.end_row,
                file_path=chunk.file_path,
                timestamp=time.time(),
                memory_usage=0  # Will be calculated
            )
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return None
    
    def batch_stream(self, file_paths: List[str],
                    batch_size: Optional[int] = None,
                    transform_func: Optional[Callable] = None,
                    **kwargs) -> Iterator[List[DataChunk]]:
        """
        Stream data in batches for more efficient processing
        """
        actual_batch_size = batch_size or self.stream_config.batch_size
        batch = []
        
        for chunk in self.stream_multiple_files(file_paths, transform_func, **kwargs):
            batch.append(chunk)
            
            if len(batch) >= actual_batch_size:
                yield batch
                batch = []
        
        # Yield remaining batch
        if batch:
            yield batch
    
    def windowed_stream(self, file_path: str, window_size: int,
                       step_size: Optional[int] = None,
                       transform_func: Optional[Callable] = None,
                       **kwargs) -> Iterator[DataChunk]:
        """
        Stream data in sliding windows
        """
        step_size = step_size or window_size
        window_buffer = []
        
        for chunk in self.stream_file(file_path, transform_func, **kwargs):
            window_buffer.append(chunk.data)
            
            # Check if window is full
            total_rows = sum(len(df) for df in window_buffer)
            if total_rows >= window_size:
                # Combine data in window
                combined_data = pd.concat(window_buffer, ignore_index=True)
                
                # Create windowed chunk
                windowed_chunk = DataChunk(
                    data=combined_data.head(window_size),
                    chunk_id=chunk.chunk_id,
                    start_row=chunk.start_row,
                    end_row=chunk.start_row + window_size,
                    file_path=chunk.file_path,
                    timestamp=time.time(),
                    memory_usage=0
                )
                
                yield windowed_chunk
                
                # Slide window
                remaining_rows = total_rows - step_size
                if remaining_rows > 0:
                    # Keep remaining data for next window
                    combined_data = combined_data.tail(remaining_rows)
                    window_buffer = [combined_data]
                else:
                    window_buffer = []
    
    def time_based_stream(self, file_path: str, time_column: str,
                         window_duration: str,
                         transform_func: Optional[Callable] = None,
                         **kwargs) -> Iterator[DataChunk]:
        """
        Stream data in time-based windows
        """
        current_window_start = None
        window_buffer = []
        
        for chunk in self.stream_file(file_path, transform_func, **kwargs):
            data = chunk.data
            
            # Ensure time column is datetime
            if time_column in data.columns:
                data[time_column] = pd.to_datetime(data[time_column])
                
                # Initialize window start
                if current_window_start is None:
                    current_window_start = data[time_column].min()
                
                # Calculate window end
                window_end = current_window_start + pd.Timedelta(window_duration)
                
                # Filter data within window
                window_data = data[data[time_column] < window_end]
                remaining_data = data[data[time_column] >= window_end]
                
                if not window_data.empty:
                    window_buffer.append(window_data)
                
                # If we have data beyond the window, yield current window
                if not remaining_data.empty:
                    if window_buffer:
                        combined_data = pd.concat(window_buffer, ignore_index=True)
                        
                        window_chunk = DataChunk(
                            data=combined_data,
                            chunk_id=chunk.chunk_id,
                            start_row=chunk.start_row,
                            end_row=chunk.start_row + len(combined_data),
                            file_path=chunk.file_path,
                            timestamp=time.time(),
                            memory_usage=0
                        )
                        
                        yield window_chunk
                    
                    # Start new window
                    current_window_start = remaining_data[time_column].min()
                    window_buffer = [remaining_data]
            else:
                # No time column, just buffer the data
                window_buffer.append(data)
        
        # Yield final window if any data remains
        if window_buffer:
            combined_data = pd.concat(window_buffer, ignore_index=True)
            final_chunk = DataChunk(
                data=combined_data,
                chunk_id=999999,  # Final chunk marker
                start_row=0,
                end_row=len(combined_data),
                file_path=file_path,
                timestamp=time.time(),
                memory_usage=0
            )
            yield final_chunk
    
    def stop_streaming(self):
        """Stop the streaming process"""
        self._is_streaming = False
        logger.info("Data streaming stopped")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'is_streaming': self._is_streaming,
            'chunks_processed': self._stats.chunks_processed,
            'total_rows': self._stats.total_rows,
            'bytes_processed': self._stats.bytes_processed,
            'avg_processing_time': self._stats.get_avg_processing_time(),
            'throughput_rows_per_sec': self._stats.get_throughput(),
            'memory_usage_mb': self._memory_monitor.get_current_usage(),
            'peak_memory_mb': self._memory_monitor.get_peak_usage()
        }


class StreamingStats:
    """Track streaming performance statistics"""
    
    def __init__(self):
        self.chunks_processed = 0
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = []
        self.start_time = time.time()
    
    def update(self, chunk: DataChunk):
        """Update statistics with a processed chunk"""
        self.chunks_processed += 1
        self.total_rows += len(chunk.data)
        self.bytes_processed += chunk.data.memory_usage(deep=True).sum()
        
        # Track processing time
        processing_time = time.time() - chunk.timestamp
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times (last 100)
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per chunk"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_throughput(self) -> float:
        """Get throughput in rows per second"""
        elapsed = time.time() - self.start_time
        return self.total_rows / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.chunks_processed = 0
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = []
        self.start_time = time.time()


class MemoryMonitor:
    """Monitor memory usage during streaming"""
    
    def __init__(self, threshold_mb: float):
        self.threshold_mb = threshold_mb
        self.peak_usage = 0.0
        self.current_usage = 0.0
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is under pressure"""
        self.current_usage = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_usage = max(self.peak_usage, self.current_usage)
        return self.current_usage > self.threshold_mb
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.current_usage
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB"""
        return self.peak_usage


# Async streaming support
class AsyncDataStreamer:
    """Async version of DataStreamer for better concurrency"""
    
    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()
        self.data_loader = ScalableDataLoader(self.config)
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
    
    async def stream_file_async(self, file_path: str,
                              transform_func: Optional[Callable] = None,
                              **kwargs) -> AsyncIterator[DataChunk]:
        """Async version of stream_file"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            
            # Run blocking operations in thread pool
            chunks = await loop.run_in_executor(
                None, 
                lambda: list(self.data_loader.load_chunks(file_path, **kwargs))
            )
            
            for chunk in chunks:
                if transform_func:
                    # Run transformation in thread pool
                    transformed_data = await loop.run_in_executor(
                        None, transform_func, chunk.data
                    )
                    chunk.data = transformed_data
                
                yield chunk
    
    async def stream_multiple_files_async(self, file_paths: List[str],
                                        transform_func: Optional[Callable] = None,
                                        **kwargs) -> AsyncIterator[DataChunk]:
        """Async version of stream_multiple_files"""
        tasks = [
            self.stream_file_async(file_path, transform_func, **kwargs)
            for file_path in file_paths
        ]
        
        async for chunk in self._merge_async_streams(tasks):
            yield chunk
    
    async def _merge_async_streams(self, streams: List[AsyncIterator[DataChunk]]) -> AsyncIterator[DataChunk]:
        """Merge multiple async streams"""
        # This is a simplified implementation
        # In production, you'd want a more sophisticated merging strategy
        for stream in streams:
            async for chunk in stream:
                yield chunk


# Utility functions
def create_time_filter(start_time: str, end_time: str, 
                      time_column: str = 'timestamp') -> Callable[[pd.DataFrame], bool]:
    """Create a time-based filter function"""
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    
    def filter_func(df: pd.DataFrame) -> bool:
        if time_column not in df.columns:
            return True
        
        df_time = pd.to_datetime(df[time_column])
        return not df_time.between(start_dt, end_dt).empty
    
    return filter_func

def create_column_selector(columns: List[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a column selection transform function"""
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [col for col in columns if col in df.columns]
        return df[available_cols] if available_cols else df
    
    return transform_func

def create_sampling_transform(sample_rate: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a sampling transform function"""
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        if sample_rate >= 1.0:
            return df
        return df.sample(frac=sample_rate)
    
    return transform_func