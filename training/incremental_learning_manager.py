"""
Incremental Learning Manager for Large Dataset Training
Handles streaming data, memory-efficient training, and adaptive learning strategies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator
import logging
from collections import deque, defaultdict
import time
import gc
import psutil
import pickle
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class DataChunk:
    """Represents a chunk of training data for incremental learning"""
    data: np.ndarray
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    chunk_id: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class IncrementalLearningConfig:
    """Configuration for incremental learning"""
    # Data streaming settings
    chunk_size: int = 1024
    max_chunks_in_memory: int = 10
    prefetch_chunks: int = 2
    
    # Learning adaptation settings
    base_learning_rate: float = 3e-4
    adaptive_lr_factor: float = 0.95
    forgetting_factor: float = 0.01
    experience_replay_ratio: float = 0.2
    
    # Memory management
    memory_threshold_gb: float = 8.0
    gradient_checkpoint_layers: int = 4
    
    # Catastrophic forgetting prevention
    ewc_lambda: float = 0.4
    importance_threshold: float = 0.1
    
    # Performance optimization
    use_mixed_precision: bool = True
    parallel_data_loading: bool = True
    pin_memory: bool = True


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.fisher_information = {}
        self.optimal_weights = {}
        
    def consolidate(self, dataloader: Iterator[torch.Tensor], num_samples: int = 1000):
        """
        Compute Fisher Information Matrix and store optimal weights
        """
        logger.info("Computing Fisher Information Matrix for EWC...")
        
        # Initialize Fisher Information
        self.fisher_information = {}
        for n, p in self.params.items():
            self.fisher_information[n] = torch.zeros_like(p)
        
        # Store current optimal weights
        self.optimal_weights = {}
        for n, p in self.params.items():
            self.optimal_weights[n] = p.clone().detach()
        
        self.model.eval()
        
        # Estimate Fisher Information
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            # Forward pass
            self.model.zero_grad()
            if isinstance(batch, (list, tuple)):
                output = self.model(batch[0])
                if len(batch) > 1:
                    # If we have labels, use them for loss calculation
                    loss = nn.functional.cross_entropy(output, batch[1])
                else:
                    # For unsupervised or self-supervised learning
                    loss = output.mean()
            else:
                output = self.model(batch)
                loss = output.mean()
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher Information
            for n, p in self.params.items():
                if p.grad is not None:
                    self.fisher_information[n] += p.grad.data.clone().pow(2)
            
            sample_count += batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
        
        # Normalize Fisher Information
        for n in self.fisher_information:
            self.fisher_information[n] /= sample_count
        
        logger.info(f"EWC consolidation completed with {sample_count} samples")
    
    def penalty(self) -> torch.Tensor:
        """
        Calculate EWC penalty term
        """
        penalty = 0.0
        for n, p in self.params.items():
            if n in self.fisher_information:
                penalty += (self.fisher_information[n] * 
                           (p - self.optimal_weights[n]).pow(2)).sum()
        
        return self.lambda_ewc * penalty


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for incremental learning
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 base_lr: float = 3e-4,
                 adaptation_factor: float = 0.95,
                 patience: int = 10,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.adaptation_factor = adaptation_factor
        self.patience = patience
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.wait_count = 0
        self.learning_rates = []
        
    def step(self, loss: float):
        """
        Update learning rate based on performance
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.adaptation_factor, self.min_lr)
                param_group['lr'] = new_lr
                
            self.wait_count = 0
            logger.info(f"Adaptive LR: {old_lr:.6f} -> {new_lr:.6f}")
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        return current_lr


class IncrementalDataLoader:
    """
    Memory-efficient data loader for large datasets
    """
    
    def __init__(self, 
                 data_source: str,
                 config: IncrementalLearningConfig,
                 transform=None):
        self.data_source = data_source
        self.config = config
        self.transform = transform
        
        # Memory management
        self.chunks_queue = queue.Queue(maxsize=config.max_chunks_in_memory)
        self.prefetch_queue = queue.Queue(maxsize=config.prefetch_chunks)
        
        # Threading for parallel data loading
        self.loading_thread = None
        self.stop_loading = threading.Event()
        
        # Statistics
        self.chunks_loaded = 0
        self.total_samples = 0
        self.loading_times = deque(maxlen=100)
        
        # Initialize data source
        self._initialize_data_source()
        
    def _initialize_data_source(self):
        """Initialize data source (file, database, etc.)"""
        if os.path.exists(self.data_source):
            if self.data_source.endswith('.csv'):
                import pandas as pd
                self.data = pd.read_csv(self.data_source)
            elif self.data_source.endswith('.npy'):
                self.data = np.load(self.data_source)
            else:
                raise ValueError(f"Unsupported data source format: {self.data_source}")
        else:
            raise FileNotFoundError(f"Data source not found: {self.data_source}")
        
        self.total_samples = len(self.data)
        logger.info(f"Initialized data source with {self.total_samples} samples")
    
    def _load_chunk(self, chunk_id: int) -> DataChunk:
        """Load a single chunk of data"""
        start_time = time.time()
        
        start_idx = chunk_id * self.config.chunk_size
        end_idx = min(start_idx + self.config.chunk_size, self.total_samples)
        
        if start_idx >= self.total_samples:
            return None
        
        chunk_data = self.data[start_idx:end_idx]
        if hasattr(chunk_data, 'values'):
            chunk_data = chunk_data.values
        
        # Apply transform if provided
        if self.transform:
            chunk_data = self.transform(chunk_data)
        
        loading_time = time.time() - start_time
        self.loading_times.append(loading_time)
        
        return DataChunk(
            data=chunk_data,
            chunk_id=chunk_id,
            timestamp=start_time,
            metadata={
                'start_idx': start_idx,
                'end_idx': end_idx,
                'loading_time': loading_time
            }
        )
    
    def _background_loader(self):
        """Background thread for loading data chunks"""
        chunk_id = 0
        
        while not self.stop_loading.is_set():
            try:
                chunk = self._load_chunk(chunk_id)
                if chunk is None:
                    break
                    
                self.prefetch_queue.put(chunk, timeout=1.0)
                chunk_id += 1
                self.chunks_loaded += 1
                
            except queue.Full:
                # Queue is full, wait a bit
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in background loader: {e}")
                break
        
        logger.info(f"Background loader finished after {self.chunks_loaded} chunks")
    
    def start_loading(self):
        """Start background data loading"""
        if self.config.parallel_data_loading:
            self.loading_thread = threading.Thread(target=self._background_loader)
            self.loading_thread.daemon = True
            self.loading_thread.start()
            logger.info("Started background data loading")
    
    def stop_loading(self):
        """Stop background data loading"""
        self.stop_loading.set()
        if self.loading_thread:
            self.loading_thread.join()
    
    def __iter__(self):
        """Iterator for data chunks"""
        if self.config.parallel_data_loading:
            # Use prefetched chunks
            while True:
                try:
                    chunk = self.prefetch_queue.get(timeout=5.0)
                    yield chunk
                except queue.Empty:
                    break
        else:
            # Sequential loading
            chunk_id = 0
            while True:
                chunk = self._load_chunk(chunk_id)
                if chunk is None:
                    break
                yield chunk
                chunk_id += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        return {
            'chunks_loaded': self.chunks_loaded,
            'total_samples': self.total_samples,
            'avg_loading_time': np.mean(self.loading_times) if self.loading_times else 0,
            'max_loading_time': np.max(self.loading_times) if self.loading_times else 0,
            'memory_usage_gb': psutil.Process().memory_info().rss / 1024**3
        }


class IncrementalLearningManager:
    """
    Main manager for incremental learning on large datasets
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: IncrementalLearningConfig,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize components
        self.ewc = ElasticWeightConsolidation(self.model, config.ewc_lambda)
        self.adaptive_lr = AdaptiveLearningRateScheduler(
            optimizer, 
            config.base_learning_rate,
            config.adaptive_lr_factor
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Training statistics
        self.training_stats = {
            'chunks_processed': 0,
            'total_samples': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'learning_rates': [],
            'memory_usage': [],
            'processing_times': [],
            'forgetting_penalties': []
        }
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor(config.memory_threshold_gb)
        
        logger.info(f"Incremental Learning Manager initialized on {self.device}")
    
    def _memory_cleanup(self):
        """Perform memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _update_experience_buffer(self, chunk: DataChunk):
        """Update experience replay buffer with new data"""
        if len(self.experience_buffer) >= self.experience_buffer.maxlen:
            # Remove old experiences
            for _ in range(len(chunk.data) // 2):
                self.experience_buffer.popleft()
        
        # Add new experiences
        for i, sample in enumerate(chunk.data):
            self.experience_buffer.append({
                'data': sample,
                'timestamp': chunk.timestamp,
                'chunk_id': chunk.chunk_id,
                'sample_id': i
            })
    
    def _get_replay_batch(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get batch from experience replay buffer"""
        if len(self.experience_buffer) < batch_size:
            return None
        
        # Sample random experiences
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[idx]['data'] for idx in indices]
        
        return torch.FloatTensor(np.array(batch)).to(self.device)
    
    def _compute_loss(self, batch: torch.Tensor, is_replay: bool = False) -> torch.Tensor:
        """Compute loss for a batch"""
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.model(batch)
                # For self-supervised learning, use reconstruction loss
                loss = nn.functional.mse_loss(output, batch)
        else:
            output = self.model(batch)
            loss = nn.functional.mse_loss(output, batch)
        
        # Add EWC penalty to prevent catastrophic forgetting
        if not is_replay and hasattr(self.ewc, 'fisher_information'):
            ewc_penalty = self.ewc.penalty()
            loss += ewc_penalty
            self.training_stats['forgetting_penalties'].append(ewc_penalty.item())
        
        return loss
    
    def train_on_chunk(self, chunk: DataChunk) -> Dict[str, float]:
        """Train model on a single data chunk"""
        start_time = time.time()
        
        # Memory check
        if self.memory_monitor.check_memory():
            self._memory_cleanup()
        
        # Convert chunk to tensor
        chunk_tensor = torch.FloatTensor(chunk.data).to(self.device)
        
        # Update experience buffer
        self._update_experience_buffer(chunk)
        
        # Training step
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Process chunk in mini-batches
        batch_size = min(64, len(chunk.data))
        for i in range(0, len(chunk.data), batch_size):
            batch = chunk_tensor[i:i+batch_size]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss = self._compute_loss(batch)
            
            # Experience replay
            if self.config.experience_replay_ratio > 0 and len(self.experience_buffer) > 0:
                replay_batch_size = int(batch_size * self.config.experience_replay_ratio)
                replay_batch = self._get_replay_batch(replay_batch_size)
                
                if replay_batch is not None:
                    replay_loss = self._compute_loss(replay_batch, is_replay=True)
                    loss += replay_loss * self.config.experience_replay_ratio
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        current_lr = self.adaptive_lr.step(avg_loss)
        
        # Update statistics
        self.training_stats['chunks_processed'] += 1
        self.training_stats['total_samples'] += len(chunk.data)
        self.training_stats['total_loss'] += total_loss
        self.training_stats['avg_loss'] = (
            self.training_stats['total_loss'] / self.training_stats['chunks_processed']
        )
        self.training_stats['learning_rates'].append(current_lr)
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024**3
        self.training_stats['memory_usage'].append(memory_usage)
        
        # Processing time
        processing_time = time.time() - start_time
        self.training_stats['processing_times'].append(processing_time)
        
        return {
            'loss': avg_loss,
            'learning_rate': current_lr,
            'memory_usage_gb': memory_usage,
            'processing_time': processing_time,
            'samples_processed': len(chunk.data)
        }
    
    def train_incremental(self, 
                         data_source: str,
                         num_epochs: int = 1,
                         consolidation_frequency: int = 10) -> Dict[str, Any]:
        """
        Train model incrementally on large dataset
        """
        logger.info(f"Starting incremental training on {data_source}")
        
        # Initialize data loader
        data_loader = IncrementalDataLoader(data_source, self.config)
        data_loader.start_loading()
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                
                chunk_count = 0
                epoch_loss = 0.0
                
                # Process chunks
                for chunk in data_loader:
                    chunk_metrics = self.train_on_chunk(chunk)
                    epoch_loss += chunk_metrics['loss']
                    chunk_count += 1
                    
                    # Consolidate Fisher Information periodically
                    if chunk_count % consolidation_frequency == 0:
                        logger.info(f"Consolidating at chunk {chunk_count}")
                        # Create temporary dataloader for consolidation
                        temp_data = torch.FloatTensor(chunk.data).to(self.device)
                        temp_loader = [temp_data]
                        self.ewc.consolidate(temp_loader, num_samples=min(1000, len(chunk.data)))
                    
                    # Progress reporting
                    if chunk_count % 10 == 0:
                        logger.info(f"Chunk {chunk_count}: Loss={chunk_metrics['loss']:.4f}, "
                                   f"LR={chunk_metrics['learning_rate']:.6f}, "
                                   f"Memory={chunk_metrics['memory_usage_gb']:.2f}GB")
                
                # Epoch summary
                avg_epoch_loss = epoch_loss / chunk_count if chunk_count > 0 else 0.0
                logger.info(f"Epoch {epoch + 1} completed: "
                           f"Avg Loss={avg_epoch_loss:.4f}, "
                           f"Chunks={chunk_count}")
        
        finally:
            data_loader.stop_loading()
        
        # Final training statistics
        final_stats = self.get_training_summary()
        final_stats['data_loader_stats'] = data_loader.get_stats()
        
        return final_stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'chunks_processed': self.training_stats['chunks_processed'],
            'total_samples': self.training_stats['total_samples'],
            'final_loss': self.training_stats['avg_loss'],
            'avg_memory_usage_gb': np.mean(self.training_stats['memory_usage']) if self.training_stats['memory_usage'] else 0,
            'max_memory_usage_gb': np.max(self.training_stats['memory_usage']) if self.training_stats['memory_usage'] else 0,
            'avg_processing_time': np.mean(self.training_stats['processing_times']) if self.training_stats['processing_times'] else 0,
            'total_processing_time': np.sum(self.training_stats['processing_times']) if self.training_stats['processing_times'] else 0,
            'learning_rate_history': self.training_stats['learning_rates'],
            'experience_buffer_size': len(self.experience_buffer),
            'forgetting_penalties': self.training_stats['forgetting_penalties']
        }
    
    def save_state(self, filepath: str):
        """Save incremental learning state"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'experience_buffer': list(self.experience_buffer),
            'ewc_fisher': self.ewc.fisher_information if hasattr(self.ewc, 'fisher_information') else {},
            'ewc_weights': self.ewc.optimal_weights if hasattr(self.ewc, 'optimal_weights') else {},
            'adaptive_lr_state': {
                'best_loss': self.adaptive_lr.best_loss,
                'wait_count': self.adaptive_lr.wait_count,
                'learning_rates': self.adaptive_lr.learning_rates
            }
        }
        
        torch.save(state, filepath)
        logger.info(f"Incremental learning state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load incremental learning state"""
        state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_stats = state['training_stats']
        self.experience_buffer = deque(state['experience_buffer'], maxlen=10000)
        
        # Restore EWC state
        if 'ewc_fisher' in state and state['ewc_fisher']:
            self.ewc.fisher_information = state['ewc_fisher']
        if 'ewc_weights' in state and state['ewc_weights']:
            self.ewc.optimal_weights = state['ewc_weights']
        
        # Restore adaptive LR state
        if 'adaptive_lr_state' in state:
            lr_state = state['adaptive_lr_state']
            self.adaptive_lr.best_loss = lr_state['best_loss']
            self.adaptive_lr.wait_count = lr_state['wait_count']
            self.adaptive_lr.learning_rates = lr_state['learning_rates']
        
        logger.info(f"Incremental learning state loaded from {filepath}")


class MemoryMonitor:
    """Monitor memory usage during training"""
    
    def __init__(self, threshold_gb: float = 8.0):
        self.threshold_gb = threshold_gb
        self.memory_history = deque(maxlen=100)
        
    def check_memory(self) -> bool:
        """Check if memory usage exceeds threshold"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        self.memory_history.append(memory_gb)
        
        if memory_gb > self.threshold_gb:
            logger.warning(f"Memory usage {memory_gb:.2f}GB exceeds threshold {self.threshold_gb}GB")
            return True
        
        return False
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        if not self.memory_history:
            return {'current': 0, 'avg': 0, 'max': 0}
        
        return {
            'current': self.memory_history[-1],
            'avg': np.mean(self.memory_history),
            'max': np.max(self.memory_history)
        }


def create_incremental_learning_config(
    dataset_size_gb: float,
    available_memory_gb: float,
    target_epochs: int = 1
) -> IncrementalLearningConfig:
    """
    Create optimized incremental learning configuration based on dataset size
    """
    # Calculate optimal chunk size
    chunk_size = min(2048, int(available_memory_gb * 1024 * 1024 / dataset_size_gb))
    chunk_size = max(64, chunk_size)  # Minimum chunk size
    
    # Calculate max chunks in memory
    max_chunks = max(5, int(available_memory_gb / 2))
    
    return IncrementalLearningConfig(
        chunk_size=chunk_size,
        max_chunks_in_memory=max_chunks,
        prefetch_chunks=min(3, max_chunks // 2),
        memory_threshold_gb=available_memory_gb * 0.8,
        use_mixed_precision=True,
        parallel_data_loading=True,
        experience_replay_ratio=0.1,
        ewc_lambda=0.4,
        adaptive_lr_factor=0.98
    )