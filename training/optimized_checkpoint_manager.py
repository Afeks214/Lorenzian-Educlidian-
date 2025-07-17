"""
Optimized Checkpoint Manager for Large Training Runs
Supports incremental saves, compression, cloud storage, and efficient recovery
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import time
import json
import pickle
import gzip
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import psutil
import tempfile
from enum import Enum
import asyncio
import aiofiles
import weakref

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression types for checkpoints"""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    BZIP2 = "bzip2"


class StorageBackend(Enum):
    """Storage backends for checkpoints"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HDFS = "hdfs"


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    checkpoint_id: str
    timestamp: datetime
    epoch: int
    step: int
    loss: float
    model_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    save_duration: float
    model_architecture: str
    optimizer_type: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    git_commit: Optional[str] = None
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    
    def __post_init__(self):
        import sys
        self.python_version = sys.version
        self.torch_version = torch.__version__
        if torch.cuda.is_available():
            self.cuda_version = torch.version.cuda


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint manager"""
    # Basic settings
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 100  # Save every N steps
    keep_last_n: int = 5  # Keep last N checkpoints
    keep_best_n: int = 3   # Keep best N checkpoints
    
    # Storage optimization
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    async_saving: bool = True
    incremental_saving: bool = True
    
    # Performance settings
    save_timeout: float = 300.0  # 5 minutes
    max_concurrent_saves: int = 2
    buffer_size: int = 1024 * 1024  # 1MB buffer
    
    # Backup and recovery
    backup_to_cloud: bool = False
    storage_backend: StorageBackend = StorageBackend.LOCAL
    cloud_config: Dict[str, Any] = None
    
    # Monitoring
    monitor_disk_usage: bool = True
    max_disk_usage_gb: float = 100.0
    alert_threshold_gb: float = 80.0
    
    # Recovery settings
    auto_recovery: bool = True
    recovery_timeout: float = 60.0
    max_recovery_attempts: int = 3


class IncrementalStateTracker:
    """Track incremental changes in model state"""
    
    def __init__(self):
        self.previous_state = {}
        self.state_hashes = {}
        self.change_history = []
        
    def compute_state_diff(self, current_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute difference between current and previous state"""
        diff = {
            'changed_params': {},
            'unchanged_params': [],
            'new_params': [],
            'removed_params': [],
            'total_changes': 0,
            'change_ratio': 0.0
        }
        
        total_params = 0
        changed_params = 0
        
        # Check for changes in existing parameters
        for name, tensor in current_state.items():
            total_params += tensor.numel()
            
            # Compute hash of current tensor
            current_hash = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()
            
            if name in self.previous_state:
                previous_hash = self.state_hashes.get(name, '')
                if current_hash != previous_hash:
                    # Parameter changed
                    diff['changed_params'][name] = {
                        'tensor': tensor,
                        'previous_hash': previous_hash,
                        'current_hash': current_hash
                    }
                    changed_params += tensor.numel()
                else:
                    # Parameter unchanged
                    diff['unchanged_params'].append(name)
            else:
                # New parameter
                diff['new_params'].append(name)
                diff['changed_params'][name] = {
                    'tensor': tensor,
                    'previous_hash': None,
                    'current_hash': current_hash
                }
                changed_params += tensor.numel()
            
            # Update hash
            self.state_hashes[name] = current_hash
        
        # Check for removed parameters
        for name in self.previous_state:
            if name not in current_state:
                diff['removed_params'].append(name)
        
        # Calculate change statistics
        diff['total_changes'] = changed_params
        diff['change_ratio'] = changed_params / total_params if total_params > 0 else 0.0
        
        # Update previous state
        self.previous_state = {name: tensor.clone() for name, tensor in current_state.items()}
        
        return diff
    
    def should_save_incremental(self, change_ratio: float, threshold: float = 0.01) -> bool:
        """Determine if incremental save is worthwhile"""
        return change_ratio > threshold


class AsyncCheckpointSaver:
    """Asynchronous checkpoint saving"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.save_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_saves)
        self.active_saves = {}
        self.save_history = []
        
    def save_async(self, 
                   checkpoint_data: Dict[str, Any],
                   filepath: str,
                   metadata: CheckpointMetadata) -> str:
        """Save checkpoint asynchronously"""
        save_id = f"{metadata.checkpoint_id}_{int(time.time())}"
        
        # Submit save task
        future = self.executor.submit(
            self._save_checkpoint_sync,
            checkpoint_data,
            filepath,
            metadata
        )
        
        self.active_saves[save_id] = {
            'future': future,
            'filepath': filepath,
            'metadata': metadata,
            'start_time': time.time()
        }
        
        return save_id
    
    def _save_checkpoint_sync(self, 
                             checkpoint_data: Dict[str, Any],
                             filepath: str,
                             metadata: CheckpointMetadata) -> bool:
        """Synchronous checkpoint saving implementation"""
        try:
            start_time = time.time()
            
            # Create temporary file
            temp_path = filepath + ".tmp"
            
            # Save to temporary file
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(temp_path, 'wb', compresslevel=self.config.compression_level) as f:
                    torch.save(checkpoint_data, f)
            elif self.config.compression == CompressionType.LZMA:
                import lzma
                with lzma.open(temp_path, 'wb', preset=self.config.compression_level) as f:
                    torch.save(checkpoint_data, f)
            elif self.config.compression == CompressionType.BZIP2:
                import bz2
                with bz2.open(temp_path, 'wb', compresslevel=self.config.compression_level) as f:
                    torch.save(checkpoint_data, f)
            else:
                torch.save(checkpoint_data, temp_path)
            
            # Atomic move
            shutil.move(temp_path, filepath)
            
            # Update metadata
            save_duration = time.time() - start_time
            metadata.save_duration = save_duration
            
            # Save metadata
            metadata_path = filepath.replace('.pt', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {filepath} (took {save_duration:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving checkpoint {filepath}: {e}")
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def wait_for_saves(self, timeout: float = None) -> Dict[str, bool]:
        """Wait for all active saves to complete"""
        results = {}
        
        for save_id, save_info in self.active_saves.items():
            try:
                result = save_info['future'].result(timeout=timeout)
                results[save_id] = result
                
                # Record save history
                self.save_history.append({
                    'save_id': save_id,
                    'filepath': save_info['filepath'],
                    'duration': time.time() - save_info['start_time'],
                    'success': result
                })
                
            except Exception as e:
                logger.error(f"Save {save_id} failed: {e}")
                results[save_id] = False
        
        # Clear active saves
        self.active_saves.clear()
        
        return results
    
    def get_save_stats(self) -> Dict[str, Any]:
        """Get save statistics"""
        if not self.save_history:
            return {'total_saves': 0, 'success_rate': 0.0, 'avg_duration': 0.0}
        
        successful_saves = [s for s in self.save_history if s['success']]
        
        return {
            'total_saves': len(self.save_history),
            'successful_saves': len(successful_saves),
            'success_rate': len(successful_saves) / len(self.save_history),
            'avg_duration': np.mean([s['duration'] for s in successful_saves]) if successful_saves else 0,
            'total_duration': sum(s['duration'] for s in successful_saves)
        }


class CloudStorageManager:
    """Manage cloud storage for checkpoints"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.backend = config.storage_backend
        self.cloud_config = config.cloud_config or {}
        self.client = None
        
        if self.backend != StorageBackend.LOCAL:
            self._init_cloud_client()
    
    def _init_cloud_client(self):
        """Initialize cloud storage client"""
        if self.backend == StorageBackend.S3:
            try:
                import boto3
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.cloud_config.get('access_key'),
                    aws_secret_access_key=self.cloud_config.get('secret_key'),
                    region_name=self.cloud_config.get('region', 'us-east-1')
                )
            except ImportError:
                logger.error("boto3 required for S3 backend")
                
        elif self.backend == StorageBackend.GCS:
            try:
                from google.cloud import storage
                self.client = storage.Client(
                    project=self.cloud_config.get('project_id'),
                    credentials=self.cloud_config.get('credentials')
                )
            except ImportError:
                logger.error("google-cloud-storage required for GCS backend")
                
        elif self.backend == StorageBackend.AZURE:
            try:
                from azure.storage.blob import BlobServiceClient
                self.client = BlobServiceClient(
                    account_url=self.cloud_config.get('account_url'),
                    credential=self.cloud_config.get('credential')
                )
            except ImportError:
                logger.error("azure-storage-blob required for Azure backend")
    
    def upload_checkpoint(self, local_path: str, remote_path: str) -> bool:
        """Upload checkpoint to cloud storage"""
        try:
            if self.backend == StorageBackend.S3:
                bucket = self.cloud_config.get('bucket')
                self.client.upload_file(local_path, bucket, remote_path)
                
            elif self.backend == StorageBackend.GCS:
                bucket = self.client.bucket(self.cloud_config.get('bucket'))
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_path)
                
            elif self.backend == StorageBackend.AZURE:
                container = self.cloud_config.get('container')
                blob_client = self.client.get_blob_client(
                    container=container,
                    blob=remote_path
                )
                with open(local_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Checkpoint uploaded to cloud: {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return False
    
    def download_checkpoint(self, remote_path: str, local_path: str) -> bool:
        """Download checkpoint from cloud storage"""
        try:
            if self.backend == StorageBackend.S3:
                bucket = self.cloud_config.get('bucket')
                self.client.download_file(bucket, remote_path, local_path)
                
            elif self.backend == StorageBackend.GCS:
                bucket = self.client.bucket(self.cloud_config.get('bucket'))
                blob = bucket.blob(remote_path)
                blob.download_to_filename(local_path)
                
            elif self.backend == StorageBackend.AZURE:
                container = self.cloud_config.get('container')
                blob_client = self.client.get_blob_client(
                    container=container,
                    blob=remote_path
                )
                with open(local_path, 'wb') as data:
                    blob_client.download_blob().readinto(data)
            
            logger.info(f"Checkpoint downloaded from cloud: {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud download failed: {e}")
            return False


class OptimizedCheckpointManager:
    """
    Optimized checkpoint manager for large training runs
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.incremental_tracker = IncrementalStateTracker()
        self.async_saver = AsyncCheckpointSaver(config)
        self.cloud_manager = CloudStorageManager(config) if config.backup_to_cloud else None
        
        # Checkpoint registry
        self.checkpoint_registry = {}
        self.best_checkpoints = {}
        self.checkpoint_history = []
        
        # Performance monitoring
        self.save_times = []
        self.compression_ratios = []
        self.disk_usage_history = []
        
        # Load existing registry
        self._load_checkpoint_registry()
        
        logger.info(f"Optimized checkpoint manager initialized: {self.checkpoint_dir}")
    
    def _load_checkpoint_registry(self):
        """Load existing checkpoint registry"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_registry = data.get('registry', {})
                    self.best_checkpoints = data.get('best_checkpoints', {})
                    self.checkpoint_history = data.get('history', [])
                logger.info(f"Loaded checkpoint registry with {len(self.checkpoint_registry)} entries")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint registry: {e}")
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        registry_data = {
            'registry': self.checkpoint_registry,
            'best_checkpoints': self.best_checkpoints,
            'history': self.checkpoint_history,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get checkpoint file path"""
        return self.checkpoint_dir / f"{checkpoint_id}.pt"
    
    def _calculate_file_sizes(self, filepath: str) -> Tuple[float, float]:
        """Calculate original and compressed file sizes"""
        if os.path.exists(filepath):
            compressed_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            # Estimate original size (rough approximation)
            if self.config.compression != CompressionType.NONE:
                original_size = compressed_size * 3  # Rough estimate
            else:
                original_size = compressed_size
            
            return original_size, compressed_size
        
        return 0.0, 0.0
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints based on retention policy"""
        if len(self.checkpoint_history) <= self.config.keep_last_n:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        # Keep best checkpoints
        best_checkpoint_ids = set(self.best_checkpoints.values())
        
        # Identify checkpoints to remove
        to_remove = []
        for checkpoint in sorted_checkpoints[self.config.keep_last_n:]:
            if checkpoint['checkpoint_id'] not in best_checkpoint_ids:
                to_remove.append(checkpoint)
        
        # Remove old checkpoints
        for checkpoint in to_remove:
            checkpoint_id = checkpoint['checkpoint_id']
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            
            try:
                if checkpoint_path.exists():
                    os.remove(checkpoint_path)
                
                # Remove metadata
                metadata_path = checkpoint_path.with_suffix('.json')
                if metadata_path.exists():
                    os.remove(metadata_path)
                
                # Remove from registry
                if checkpoint_id in self.checkpoint_registry:
                    del self.checkpoint_registry[checkpoint_id]
                
                # Remove from history
                self.checkpoint_history = [
                    cp for cp in self.checkpoint_history 
                    if cp['checkpoint_id'] != checkpoint_id
                ]
                
                logger.info(f"Cleaned up old checkpoint: {checkpoint_id}")
                
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_id}: {e}")
    
    def _monitor_disk_usage(self):
        """Monitor disk usage"""
        if not self.config.monitor_disk_usage:
            return
        
        try:
            usage = psutil.disk_usage(str(self.checkpoint_dir))
            used_gb = usage.used / (1024**3)
            total_gb = usage.total / (1024**3)
            
            self.disk_usage_history.append({
                'timestamp': datetime.now().isoformat(),
                'used_gb': used_gb,
                'total_gb': total_gb,
                'usage_percent': (used_gb / total_gb) * 100
            })
            
            if used_gb > self.config.alert_threshold_gb:
                logger.warning(f"Disk usage warning: {used_gb:.2f}GB used")
            
            if used_gb > self.config.max_disk_usage_gb:
                logger.error(f"Disk usage critical: {used_gb:.2f}GB used")
                # Force cleanup
                self._cleanup_old_checkpoints()
                
        except Exception as e:
            logger.error(f"Disk usage monitoring failed: {e}")
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       step: int,
                       loss: float,
                       metrics: Dict[str, float] = None,
                       is_best: bool = False,
                       checkpoint_id: str = None) -> str:
        """
        Save optimized checkpoint
        """
        start_time = time.time()
        
        # Generate checkpoint ID
        if checkpoint_id is None:
            checkpoint_id = f"epoch_{epoch}_step_{step}_{int(time.time())}"
        
        # Monitor disk usage
        self._monitor_disk_usage()
        
        # Prepare checkpoint data
        model_state = model.state_dict()
        
        # Check if incremental saving is beneficial
        if self.config.incremental_saving:
            state_diff = self.incremental_tracker.compute_state_diff(model_state)
            if not self.incremental_tracker.should_save_incremental(state_diff['change_ratio']):
                logger.info(f"Skipping checkpoint {checkpoint_id} - minimal changes ({state_diff['change_ratio']:.4f})")
                return None
        
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics or {},
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            epoch=epoch,
            step=step,
            loss=loss,
            model_size_mb=0.0,  # Will be calculated
            compressed_size_mb=0.0,  # Will be calculated
            compression_ratio=0.0,  # Will be calculated
            save_duration=0.0,  # Will be calculated
            model_architecture=type(model).__name__,
            optimizer_type=type(optimizer).__name__,
            hyperparameters={
                'learning_rate': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            },
            performance_metrics=metrics or {}
        )
        
        # Get checkpoint path
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        # Save checkpoint
        if self.config.async_saving:
            save_id = self.async_saver.save_async(
                checkpoint_data,
                str(checkpoint_path),
                metadata
            )
        else:
            success = self.async_saver._save_checkpoint_sync(
                checkpoint_data,
                str(checkpoint_path),
                metadata
            )
            if not success:
                logger.error(f"Failed to save checkpoint {checkpoint_id}")
                return None
        
        # Calculate file sizes
        original_size, compressed_size = self._calculate_file_sizes(str(checkpoint_path))
        metadata.model_size_mb = original_size
        metadata.compressed_size_mb = compressed_size
        metadata.compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0.0
        
        # Update registry
        self.checkpoint_registry[checkpoint_id] = {
            'filepath': str(checkpoint_path),
            'metadata': asdict(metadata),
            'is_best': is_best,
            'save_time': time.time() - start_time
        }
        
        # Update best checkpoints
        if is_best:
            self.best_checkpoints[f'best_loss'] = checkpoint_id
        
        # Update history
        self.checkpoint_history.append({
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'is_best': is_best
        })
        
        # Save registry
        self._save_checkpoint_registry()
        
        # Performance tracking
        save_time = time.time() - start_time
        self.save_times.append(save_time)
        if metadata.compression_ratio > 0:
            self.compression_ratios.append(metadata.compression_ratio)
        
        # Upload to cloud if configured
        if self.cloud_manager:
            cloud_path = f"checkpoints/{checkpoint_id}.pt"
            self.cloud_manager.upload_checkpoint(str(checkpoint_path), cloud_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_id} (took {save_time:.2f}s)")
        
        return checkpoint_id
    
    def load_checkpoint(self, 
                       checkpoint_id: str,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       map_location: str = None) -> Dict[str, Any]:
        """
        Load optimized checkpoint
        """
        start_time = time.time()
        
        # Get checkpoint info
        if checkpoint_id not in self.checkpoint_registry:
            # Try to find by best model
            if checkpoint_id in self.best_checkpoints:
                checkpoint_id = self.best_checkpoints[checkpoint_id]
            else:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_info = self.checkpoint_registry[checkpoint_id]
        checkpoint_path = checkpoint_info['filepath']
        
        # Download from cloud if not available locally
        if not os.path.exists(checkpoint_path) and self.cloud_manager:
            cloud_path = f"checkpoints/{checkpoint_id}.pt"
            self.cloud_manager.download_checkpoint(cloud_path, checkpoint_path)
        
        # Load checkpoint
        try:
            if self.config.compression == CompressionType.GZIP:
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location=map_location)
            elif self.config.compression == CompressionType.LZMA:
                import lzma
                with lzma.open(checkpoint_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location=map_location)
            elif self.config.compression == CompressionType.BZIP2:
                import bz2
                with bz2.open(checkpoint_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location=map_location)
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            load_time = time.time() - start_time
            logger.info(f"Checkpoint loaded: {checkpoint_id} (took {load_time:.2f}s)")
            
            return {
                'epoch': checkpoint_data['epoch'],
                'step': checkpoint_data['step'],
                'loss': checkpoint_data['loss'],
                'metrics': checkpoint_data.get('metrics', {}),
                'load_time': load_time,
                'checkpoint_info': checkpoint_info
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise
    
    def get_best_checkpoint(self, metric: str = 'best_loss') -> Optional[str]:
        """Get best checkpoint ID for specified metric"""
        return self.best_checkpoints.get(metric)
    
    def list_checkpoints(self, 
                        limit: int = None,
                        sort_by: str = 'timestamp') -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = list(self.checkpoint_registry.values())
        
        # Sort checkpoints
        if sort_by == 'timestamp':
            checkpoints.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        elif sort_by == 'loss':
            checkpoints.sort(key=lambda x: x['metadata']['loss'])
        elif sort_by == 'epoch':
            checkpoints.sort(key=lambda x: x['metadata']['epoch'], reverse=True)
        
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        stats = {
            'total_checkpoints': len(self.checkpoint_registry),
            'best_checkpoints': len(self.best_checkpoints),
            'total_size_mb': sum(
                cp['metadata']['compressed_size_mb'] 
                for cp in self.checkpoint_registry.values()
            ),
            'avg_save_time': np.mean(self.save_times) if self.save_times else 0,
            'avg_compression_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 0,
            'disk_usage': self.disk_usage_history[-1] if self.disk_usage_history else None
        }
        
        # Add async saver stats
        stats['async_saver'] = self.async_saver.get_save_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        # Wait for pending saves
        if self.async_saver:
            self.async_saver.wait_for_saves(timeout=self.config.save_timeout)
        
        # Save final registry
        self._save_checkpoint_registry()
        
        logger.info("Checkpoint manager cleanup completed")


def create_checkpoint_config(
    checkpoint_dir: str,
    compression_enabled: bool = True,
    cloud_backup: bool = False,
    storage_backend: str = "local"
) -> CheckpointConfig:
    """Create optimized checkpoint configuration"""
    
    # Determine compression based on available disk space
    compression = CompressionType.GZIP if compression_enabled else CompressionType.NONE
    
    # Convert storage backend string to enum
    backend_map = {
        'local': StorageBackend.LOCAL,
        's3': StorageBackend.S3,
        'gcs': StorageBackend.GCS,
        'azure': StorageBackend.AZURE
    }
    
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        compression=compression,
        async_saving=True,
        incremental_saving=True,
        backup_to_cloud=cloud_backup,
        storage_backend=backend_map.get(storage_backend, StorageBackend.LOCAL),
        monitor_disk_usage=True,
        auto_recovery=True
    )
    
    return config