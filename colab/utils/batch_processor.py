"""
Efficient Batch Processing Framework for Large Dataset Training

This module provides memory-efficient batch processing capabilities for training
MAPPO agents on large datasets without loading everything into memory at once.
"""

import numpy as np
import pandas as pd
import torch
import gc
from typing import Iterator, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 32
    sequence_length: int = 100
    overlap: int = 20
    prefetch_batches: int = 2
    max_memory_percent: float = 80.0
    checkpoint_frequency: int = 100
    enable_caching: bool = True
    cache_size: int = 1000
    num_workers: int = 2


class MemoryMonitor:
    """Monitor memory usage and optimize batch sizes"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_history = deque(maxlen=100)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        
        usage = {
            'system_used_gb': memory_info.rss / 1024**3,
            'system_total_gb': system_memory.total / 1024**3,
            'system_percent': system_memory.percent,
            'process_percent': process.memory_percent(),
            **gpu_memory
        }
        
        self.memory_history.append(usage)
        return usage
    
    def optimize_batch_size(self, current_batch_size: int, target_memory_percent: float = None) -> int:
        """Optimize batch size based on current memory usage"""
        if target_memory_percent is None:
            target_memory_percent = self.max_memory_percent
            
        usage = self.get_memory_usage()
        current_percent = usage['system_percent']
        
        if current_percent > target_memory_percent:
            # Reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size} (memory: {current_percent:.1f}%)")
            return new_batch_size
        elif current_percent < target_memory_percent * 0.7:
            # Increase batch size
            new_batch_size = min(current_batch_size * 2, int(current_batch_size * 1.2))
            logger.info(f"Increasing batch size from {current_batch_size} to {new_batch_size} (memory: {current_percent:.1f}%)")
            return new_batch_size
        
        return current_batch_size
    
    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SlidingWindowDataLoader:
    """Efficient sliding window data loader for time series"""
    
    def __init__(self, 
                 data_path: str, 
                 config: BatchConfig,
                 chunksize: int = 10000):
        self.data_path = data_path
        self.config = config
        self.chunksize = chunksize
        self.memory_monitor = MemoryMonitor(config.max_memory_percent)
        self.cache = {} if config.enable_caching else None
        self.cache_keys = deque(maxlen=config.cache_size)
        
    def _load_data_chunk(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a chunk of data from the specified range"""
        cache_key = f"{start_idx}_{end_idx}"
        
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Read specific rows from CSV
        df = pd.read_csv(self.data_path, skiprows=range(1, start_idx + 1), nrows=end_idx - start_idx)
        
        # Cache the chunk if caching is enabled
        if self.cache and len(self.cache_keys) < self.config.cache_size:
            self.cache[cache_key] = df
            self.cache_keys.append(cache_key)
        elif self.cache:
            # Remove oldest cached chunk
            oldest_key = self.cache_keys.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            self.cache[cache_key] = df
            self.cache_keys.append(cache_key)
        
        return df
    
    def get_dataset_size(self) -> int:
        """Get total number of rows in the dataset"""
        # Use pandas to get row count efficiently
        df_head = pd.read_csv(self.data_path, nrows=1)
        with open(self.data_path, 'r') as f:
            return sum(1 for _ in f) - 1  # Subtract header
    
    def create_sliding_windows(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Create sliding windows from the dataset"""
        if end_idx is None:
            end_idx = self.get_dataset_size()
        
        window_size = self.config.sequence_length
        step_size = window_size - self.config.overlap
        
        current_idx = start_idx
        while current_idx + window_size <= end_idx:
            # Load chunk containing the window
            chunk_start = max(0, current_idx - 100)  # Buffer for technical indicators
            chunk_end = min(end_idx, current_idx + window_size + 100)
            
            chunk = self._load_data_chunk(chunk_start, chunk_end)
            
            # Extract the actual window
            window_start = current_idx - chunk_start
            window_end = window_start + window_size
            
            if window_end <= len(chunk):
                window = chunk.iloc[window_start:window_end].copy()
                yield window
            
            current_idx += step_size
    
    def create_batches(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Iterator[List[pd.DataFrame]]:
        """Create batches of sliding windows"""
        batch = []
        batch_size = self.config.batch_size
        
        for window in self.create_sliding_windows(start_idx, end_idx):
            batch.append(window)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
                
                # Adaptive batch size based on memory
                new_batch_size = self.memory_monitor.optimize_batch_size(batch_size)
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    self.config.batch_size = batch_size
                
                # Clean up memory periodically
                if len(batch) % 10 == 0:
                    self.memory_monitor.cleanup_memory()
        
        # Yield remaining batch if not empty
        if batch:
            yield batch


class DataStreamer:
    """Efficient data streaming for continuous training"""
    
    def __init__(self, data_loader: SlidingWindowDataLoader, prefetch_size: int = 2):
        self.data_loader = data_loader
        self.prefetch_size = prefetch_size
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.streaming = False
        
    def start_streaming(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Start streaming data in background"""
        self.streaming = True
        self.batch_generator = self.data_loader.create_batches(start_idx, end_idx)
        
        # Prefetch initial batches
        for _ in range(self.prefetch_size):
            try:
                batch = next(self.batch_generator)
                self.prefetch_queue.append(batch)
            except StopIteration:
                break
    
    def get_next_batch(self) -> Optional[List[pd.DataFrame]]:
        """Get next batch from the stream"""
        if not self.streaming or not self.prefetch_queue:
            return None
        
        # Get next batch from prefetch queue
        batch = self.prefetch_queue.popleft()
        
        # Prefetch next batch in background
        if self.streaming:
            future = self.executor.submit(self._prefetch_next_batch)
        
        return batch
    
    def _prefetch_next_batch(self):
        """Prefetch next batch"""
        try:
            batch = next(self.batch_generator)
            if len(self.prefetch_queue) < self.prefetch_size:
                self.prefetch_queue.append(batch)
        except StopIteration:
            self.streaming = False
    
    def stop_streaming(self):
        """Stop streaming"""
        self.streaming = False
        self.executor.shutdown(wait=True)


class CheckpointManager:
    """Manage training checkpoints for large datasets"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_info = {}
        self.load_checkpoint_info()
        
    def load_checkpoint_info(self):
        """Load checkpoint information"""
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.checkpoint_info = json.load(f)
    
    def save_checkpoint_info(self):
        """Save checkpoint information"""
        info_file = self.checkpoint_dir / "checkpoint_info.json"
        with open(info_file, 'w') as f:
            json.dump(self.checkpoint_info, f, indent=2)
    
    def save_checkpoint(self, 
                       model_state: Dict[str, Any], 
                       optimizer_state: Dict[str, Any],
                       epoch: int,
                       batch_idx: int,
                       data_position: int,
                       metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'data_position': data_position,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Update checkpoint info
        self.checkpoint_info[checkpoint_name] = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'data_position': data_position,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        self.save_checkpoint_info()
        
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_name: str = None) -> Optional[Dict[str, Any]]:
        """Load training checkpoint"""
        if checkpoint_name is None:
            # Load latest checkpoint
            if not self.checkpoint_info:
                return None
            checkpoint_name = max(self.checkpoint_info.keys(), 
                                key=lambda x: self.checkpoint_info[x]['timestamp'])
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_name}")
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints"""
        if len(self.checkpoint_info) > self.max_checkpoints:
            # Sort by timestamp and remove oldest
            sorted_checkpoints = sorted(self.checkpoint_info.items(), 
                                      key=lambda x: x[1]['timestamp'])
            
            for checkpoint_name, _ in sorted_checkpoints[:-self.max_checkpoints]:
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                del self.checkpoint_info[checkpoint_name]
    
    def get_latest_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the latest checkpoint"""
        if not self.checkpoint_info:
            return None
        
        latest_name = max(self.checkpoint_info.keys(), 
                         key=lambda x: self.checkpoint_info[x]['timestamp'])
        return self.checkpoint_info[latest_name]


class BatchProcessor:
    """Main batch processing coordinator"""
    
    def __init__(self, 
                 data_path: str, 
                 config: BatchConfig,
                 checkpoint_dir: str = None):
        self.data_path = data_path
        self.config = config
        self.data_loader = SlidingWindowDataLoader(data_path, config)
        self.streamer = DataStreamer(self.data_loader, config.prefetch_batches)
        self.memory_monitor = MemoryMonitor(config.max_memory_percent)
        
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = None
        
        self.batch_count = 0
        self.start_time = time.time()
        
    def process_batches(self, 
                       trainer,
                       start_idx: int = 0, 
                       end_idx: Optional[int] = None,
                       resume_from_checkpoint: bool = True) -> Iterator[Dict[str, Any]]:
        """Process batches with training"""
        
        # Resume from checkpoint if available
        if resume_from_checkpoint and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                start_idx = checkpoint['data_position']
                self.batch_count = checkpoint['batch_idx']
                logger.info(f"Resuming from checkpoint at position {start_idx}")
        
        # Start streaming
        self.streamer.start_streaming(start_idx, end_idx)
        
        try:
            while True:
                batch = self.streamer.get_next_batch()
                if batch is None:
                    break
                
                # Process the batch
                batch_start_time = time.time()
                
                # Train on the batch
                batch_metrics = self._process_batch(trainer, batch)
                
                batch_time = time.time() - batch_start_time
                self.batch_count += 1
                
                # Memory monitoring
                memory_usage = self.memory_monitor.get_memory_usage()
                
                # Yield batch results
                yield {
                    'batch_idx': self.batch_count,
                    'batch_size': len(batch),
                    'batch_time': batch_time,
                    'memory_usage': memory_usage,
                    'metrics': batch_metrics
                }
                
                # Checkpoint saving
                if (self.checkpoint_manager and 
                    self.batch_count % self.config.checkpoint_frequency == 0):
                    self._save_checkpoint(trainer, batch_metrics)
                
                # Memory cleanup
                if self.batch_count % 10 == 0:
                    self.memory_monitor.cleanup_memory()
                
        finally:
            self.streamer.stop_streaming()
    
    def _process_batch(self, trainer, batch: List[pd.DataFrame]) -> Dict[str, float]:
        """Process a single batch"""
        total_reward = 0.0
        total_loss = 0.0
        num_episodes = len(batch)
        
        for window in batch:
            # Process each window as an episode
            episode_reward, episode_loss = self._process_window(trainer, window)
            total_reward += episode_reward
            total_loss += episode_loss
        
        return {
            'avg_reward': total_reward / num_episodes,
            'avg_loss': total_loss / num_episodes,
            'num_episodes': num_episodes
        }
    
    def _process_window(self, trainer, window: pd.DataFrame) -> Tuple[float, float]:
        """Process a single window (episode)"""
        # This is a simplified version - actual implementation depends on the trainer
        episode_reward = 0.0
        episode_loss = 0.0
        
        # Check if trainer has process_batch method (for strategic trainers)
        if hasattr(trainer, 'process_batch'):
            batch_result = trainer.process_batch([window])
            return batch_result['avg_reward'], batch_result['avg_loss']
        
        # Convert window to training data
        states = self._window_to_states(window)
        
        # Simulate training step
        for state in states:
            # Get action from trainer
            if hasattr(trainer, 'get_action'):
                action = trainer.get_action(state, deterministic=False)
            else:
                action = np.random.randint(0, 5)  # Default action
            
            # Calculate reward (simplified)
            reward = np.random.normal(0, 1)  # Placeholder
            episode_reward += reward
            
            # Store transition for training
            next_state = state  # Simplified
            done = False
            if hasattr(trainer, 'store_transition'):
                trainer.store_transition(state, action, reward, next_state, done)
        
        # Update trainer
        if hasattr(trainer, 'update'):
            loss = trainer.update()
            episode_loss = loss if loss is not None else 0.0
        
        return episode_reward, episode_loss
    
    def _window_to_states(self, window: pd.DataFrame) -> List[np.ndarray]:
        """Convert window to states for training"""
        states = []
        
        for i in range(len(window)):
            # Simplified state extraction
            if i < 20:  # Need enough history
                continue
                
            # Extract features
            close_prices = window['Close'].iloc[i-20:i].values
            volumes = window['Volume'].iloc[i-20:i].values
            
            # Calculate technical indicators
            price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
            volatility = np.std(close_prices) / np.mean(close_prices)
            volume_avg = np.mean(volumes)
            
            # Create state vector
            state = np.array([price_change, volatility, volume_avg / 100000])
            states.append(state)
        
        return states
    
    def _save_checkpoint(self, trainer, metrics: Dict[str, float]):
        """Save training checkpoint"""
        if not self.checkpoint_manager:
            return
        
        # Get model and optimizer states
        model_state = trainer.get_model_state() if hasattr(trainer, 'get_model_state') else {}
        optimizer_state = trainer.get_optimizer_state() if hasattr(trainer, 'get_optimizer_state') else {}
        
        self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=0,  # Simplified
            batch_idx=self.batch_count,
            data_position=self.batch_count * self.config.batch_size,
            metrics=metrics
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_batches': self.batch_count,
            'elapsed_time': elapsed_time,
            'batches_per_second': self.batch_count / elapsed_time if elapsed_time > 0 else 0,
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'config': self.config.__dict__
        }


def calculate_optimal_batch_size(data_size: int, 
                               memory_limit_gb: float = 4.0,
                               sequence_length: int = 100) -> int:
    """Calculate optimal batch size based on dataset size and memory constraints"""
    
    # Estimate memory per sample (very rough)
    memory_per_sample = sequence_length * 0.001  # GB per sample
    
    # Maximum batch size based on memory
    max_batch_size = int(memory_limit_gb / memory_per_sample)
    
    # Adjust based on dataset size
    if data_size < 1000:
        return min(max_batch_size, 8)
    elif data_size < 10000:
        return min(max_batch_size, 16)
    elif data_size < 100000:
        return min(max_batch_size, 32)
    else:
        return min(max_batch_size, 64)


def create_large_dataset_simulation(output_path: str, 
                                  num_rows: int = 100000,
                                  features: List[str] = None) -> str:
    """Create a simulated large dataset for testing"""
    
    if features is None:
        features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    logger.info(f"Creating simulated dataset with {num_rows} rows...")
    
    # Generate timestamps
    dates = pd.date_range('2020-01-01', periods=num_rows, freq='5min')
    
    # Generate realistic price data
    np.random.seed(42)
    initial_price = 3000.0
    price_changes = np.random.normal(0, 0.001, num_rows)
    prices = initial_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLC data
    data = {
        'Date': dates,
        'Open': prices,
        'High': prices * np.random.uniform(1.0, 1.02, num_rows),
        'Low': prices * np.random.uniform(0.98, 1.0, num_rows),
        'Close': prices * np.random.uniform(0.99, 1.01, num_rows),
        'Volume': np.random.randint(1000, 10000, num_rows)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / 1024**2  # MB
    logger.info(f"Dataset created: {output_path} ({file_size:.1f} MB)")
    
    return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create test configuration
    config = BatchConfig(
        batch_size=16,
        sequence_length=100,
        overlap=20,
        prefetch_batches=2,
        max_memory_percent=70.0,
        checkpoint_frequency=50
    )
    
    # Create simulated dataset
    test_data_path = "/tmp/test_large_dataset.csv"
    create_large_dataset_simulation(test_data_path, num_rows=10000)
    
    # Test batch processing
    processor = BatchProcessor(test_data_path, config, checkpoint_dir="/tmp/checkpoints")
    
    # Mock trainer for testing
    class MockTrainer:
        def get_action(self, state, deterministic=False):
            return np.random.randint(0, 5)
        
        def store_transition(self, state, action, reward, next_state, done):
            pass
        
        def update(self):
            return np.random.random()
    
    trainer = MockTrainer()
    
    # Process batches
    for batch_result in processor.process_batches(trainer, end_idx=1000):
        print(f"Batch {batch_result['batch_idx']}: "
              f"Reward={batch_result['metrics']['avg_reward']:.3f}, "
              f"Time={batch_result['batch_time']:.3f}s")
        
        if batch_result['batch_idx'] >= 5:  # Test with limited batches
            break
    
    print("\nProcessing stats:")
    stats = processor.get_processing_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")