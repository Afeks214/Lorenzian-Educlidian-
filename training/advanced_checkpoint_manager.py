"""
Advanced Model Checkpointing and Recovery System for MARL
Implements robust checkpointing with automatic recovery, versioning, and cloud backup
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
import hashlib
import shutil
import gzip
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import boto3
from botocore.exceptions import ClientError
import redis
import yaml
from collections import defaultdict
import weakref
import signal
import atexit

logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    # Basic settings
    checkpoint_dir: str = "checkpoints"
    checkpoint_prefix: str = "marl_checkpoint"
    max_checkpoints: int = 10
    
    # Compression and optimization
    use_compression: bool = True
    compression_level: int = 6
    incremental_checkpointing: bool = True
    
    # Versioning
    enable_versioning: bool = True
    version_strategy: str = "semantic"  # "semantic", "timestamp", "hash"
    
    # Cloud backup
    enable_cloud_backup: bool = False
    cloud_provider: str = "aws"  # "aws", "gcs", "azure"
    cloud_bucket: str = ""
    cloud_prefix: str = "marl_checkpoints"
    
    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_strategy: str = "latest"  # "latest", "best", "stable"
    recovery_timeout: int = 300  # seconds
    
    # Validation
    enable_validation: bool = True
    validation_timeout: int = 60  # seconds
    
    # Monitoring
    enable_monitoring: bool = True
    health_check_interval: int = 30  # seconds
    
    # Performance
    async_saving: bool = True
    save_frequency: int = 10  # episodes
    
    # Security
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    
    # Redis settings for distributed checkpointing
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


class CheckpointMetadata:
    """Metadata for checkpoint files"""
    
    def __init__(self, 
                 checkpoint_id: str,
                 version: str,
                 timestamp: datetime,
                 episode: int,
                 step: int,
                 metrics: Dict[str, float],
                 model_config: Dict[str, Any],
                 file_path: str,
                 file_size: int,
                 checksum: str):
        self.checkpoint_id = checkpoint_id
        self.version = version
        self.timestamp = timestamp
        self.episode = episode
        self.step = step
        self.metrics = metrics
        self.model_config = model_config
        self.file_path = file_path
        self.file_size = file_size
        self.checksum = checksum
        
        # Additional metadata
        self.creation_time = time.time()
        self.validated = False
        self.cloud_synced = False
        self.recovery_tested = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'episode': self.episode,
            'step': self.step,
            'metrics': self.metrics,
            'model_config': self.model_config,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'checksum': self.checksum,
            'creation_time': self.creation_time,
            'validated': self.validated,
            'cloud_synced': self.cloud_synced,
            'recovery_tested': self.recovery_tested
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary"""
        metadata = cls(
            checkpoint_id=data['checkpoint_id'],
            version=data['version'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            episode=data['episode'],
            step=data['step'],
            metrics=data['metrics'],
            model_config=data['model_config'],
            file_path=data['file_path'],
            file_size=data['file_size'],
            checksum=data['checksum']
        )
        
        # Restore additional metadata
        metadata.creation_time = data.get('creation_time', time.time())
        metadata.validated = data.get('validated', False)
        metadata.cloud_synced = data.get('cloud_synced', False)
        metadata.recovery_tested = data.get('recovery_tested', False)
        
        return metadata


class CheckpointValidator:
    """Validates checkpoint integrity and recoverability"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.validation_cache = {}
    
    def validate_checkpoint(self, checkpoint_path: str, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint integrity"""
        try:
            # Check if already validated
            if checkpoint_path in self.validation_cache:
                return self.validation_cache[checkpoint_path]
            
            # File existence and size
            path = Path(checkpoint_path)
            if not path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            if path.stat().st_size != metadata.file_size:
                logger.error(f"Checkpoint file size mismatch: {checkpoint_path}")
                return False
            
            # Checksum validation
            if not self._validate_checksum(checkpoint_path, metadata.checksum):
                logger.error(f"Checkpoint checksum validation failed: {checkpoint_path}")
                return False
            
            # Load and validate checkpoint content
            if not self._validate_checkpoint_content(checkpoint_path):
                logger.error(f"Checkpoint content validation failed: {checkpoint_path}")
                return False
            
            # Cache result
            self.validation_cache[checkpoint_path] = True
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation error: {e}")
            return False
    
    def _validate_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """Validate file checksum"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            actual_checksum = hasher.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Checksum validation error: {e}")
            return False
    
    def _validate_checkpoint_content(self, checkpoint_path: str) -> bool:
        """Validate checkpoint content can be loaded"""
        try:
            # Try to load checkpoint
            if self.config.use_compression:
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu')
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate required keys
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'episode', 'step']
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"Missing required key in checkpoint: {key}")
                    return False
            
            # Validate model state dict
            model_state = checkpoint['model_state_dict']
            if not isinstance(model_state, dict):
                logger.error("Invalid model state dict format")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint content validation error: {e}")
            return False
    
    def test_recovery(self, checkpoint_path: str, model_factory: Callable) -> bool:
        """Test checkpoint recovery"""
        try:
            # Create dummy model
            model = model_factory()
            optimizer = optim.Adam(model.parameters())
            
            # Load checkpoint
            checkpoint = self._load_checkpoint(checkpoint_path)
            
            # Restore model and optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Test forward pass
            dummy_input = torch.randn(1, 128)  # Adjust based on your model
            with torch.no_grad():
                output = model(dummy_input)
            
            return output is not None
            
        except Exception as e:
            logger.error(f"Recovery test error: {e}")
            return False
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint with appropriate decompression"""
        if self.config.use_compression:
            with gzip.open(checkpoint_path, 'rb') as f:
                return torch.load(f, map_location='cpu')
        else:
            return torch.load(checkpoint_path, map_location='cpu')


class CloudBackupManager:
    """Manages cloud backup of checkpoints"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.client = None
        
        if config.enable_cloud_backup:
            self._initialize_cloud_client()
    
    def _initialize_cloud_client(self):
        """Initialize cloud storage client"""
        try:
            if self.config.cloud_provider == "aws":
                self.client = boto3.client('s3')
            elif self.config.cloud_provider == "gcs":
                # Initialize GCS client
                pass
            elif self.config.cloud_provider == "azure":
                # Initialize Azure client
                pass
            else:
                raise ValueError(f"Unsupported cloud provider: {self.config.cloud_provider}")
                
        except Exception as e:
            logger.error(f"Cloud client initialization failed: {e}")
            self.client = None
    
    def upload_checkpoint(self, checkpoint_path: str, metadata: CheckpointMetadata) -> bool:
        """Upload checkpoint to cloud storage"""
        if not self.client:
            return False
        
        try:
            # Generate cloud key
            cloud_key = f"{self.config.cloud_prefix}/{metadata.checkpoint_id}.pt"
            
            # Upload file
            if self.config.cloud_provider == "aws":
                self.client.upload_file(
                    checkpoint_path,
                    self.config.cloud_bucket,
                    cloud_key
                )
            
            # Upload metadata
            metadata_key = f"{self.config.cloud_prefix}/{metadata.checkpoint_id}_metadata.json"
            metadata_content = json.dumps(metadata.to_dict(), indent=2)
            
            if self.config.cloud_provider == "aws":
                self.client.put_object(
                    Bucket=self.config.cloud_bucket,
                    Key=metadata_key,
                    Body=metadata_content
                )
            
            logger.info(f"Checkpoint uploaded to cloud: {cloud_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return False
    
    def download_checkpoint(self, checkpoint_id: str, local_path: str) -> bool:
        """Download checkpoint from cloud storage"""
        if not self.client:
            return False
        
        try:
            cloud_key = f"{self.config.cloud_prefix}/{checkpoint_id}.pt"
            
            if self.config.cloud_provider == "aws":
                self.client.download_file(
                    self.config.cloud_bucket,
                    cloud_key,
                    local_path
                )
            
            logger.info(f"Checkpoint downloaded from cloud: {cloud_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud download failed: {e}")
            return False
    
    def list_cloud_checkpoints(self) -> List[str]:
        """List available checkpoints in cloud storage"""
        if not self.client:
            return []
        
        try:
            if self.config.cloud_provider == "aws":
                response = self.client.list_objects_v2(
                    Bucket=self.config.cloud_bucket,
                    Prefix=self.config.cloud_prefix
                )
                
                checkpoint_ids = []
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.pt') and not key.endswith('_metadata.json'):
                        checkpoint_id = Path(key).stem
                        checkpoint_ids.append(checkpoint_id)
                
                return checkpoint_ids
            
        except Exception as e:
            logger.error(f"Cloud listing failed: {e}")
            return []


class DistributedCheckpointCoordinator:
    """Coordinates checkpointing across distributed training"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.redis_client = None
        
        if config.enable_monitoring:
            self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    def register_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata):
        """Register checkpoint in distributed registry"""
        if not self.redis_client:
            return
        
        try:
            # Store checkpoint metadata
            key = f"checkpoint:{checkpoint_id}"
            self.redis_client.hset(key, mapping=metadata.to_dict())
            
            # Add to checkpoint list
            self.redis_client.lpush("checkpoint_list", checkpoint_id)
            
            # Set expiration
            self.redis_client.expire(key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Checkpoint registration failed: {e}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint ID"""
        if not self.redis_client:
            return None
        
        try:
            checkpoint_ids = self.redis_client.lrange("checkpoint_list", 0, 0)
            return checkpoint_ids[0] if checkpoint_ids else None
            
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    def coordinate_save(self, node_id: str, checkpoint_id: str) -> bool:
        """Coordinate distributed checkpoint save"""
        if not self.redis_client:
            return True  # Proceed without coordination
        
        try:
            # Use Redis lock for coordination
            lock_key = f"checkpoint_lock:{checkpoint_id}"
            lock = self.redis_client.lock(lock_key, timeout=60)
            
            if lock.acquire(blocking=True):
                try:
                    # Check if checkpoint already exists
                    if not self.redis_client.exists(f"checkpoint:{checkpoint_id}"):
                        return True  # Proceed with save
                    else:
                        logger.info(f"Checkpoint {checkpoint_id} already exists")
                        return False  # Skip save
                finally:
                    lock.release()
            else:
                logger.warning(f"Failed to acquire lock for checkpoint {checkpoint_id}")
                return False
                
        except Exception as e:
            logger.error(f"Coordination error: {e}")
            return True  # Proceed without coordination


class AdvancedCheckpointManager:
    """Advanced checkpoint manager with recovery, validation, and cloud backup"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = CheckpointValidator(config)
        self.cloud_manager = CloudBackupManager(config) if config.enable_cloud_backup else None
        self.coordinator = DistributedCheckpointCoordinator(config)
        
        # Checkpoint registry
        self.checkpoint_registry = {}
        self.load_checkpoint_registry()
        
        # Version management
        self.version_counter = 0
        self.load_version_counter()
        
        # Background tasks
        self.background_tasks = []
        self.shutdown_event = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self.cleanup)
        
        # Start background monitoring
        if config.enable_monitoring:
            self._start_monitoring()
        
        logger.info(f"Advanced checkpoint manager initialized: {config.checkpoint_dir}")
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       episode: int,
                       step: int,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint with full metadata and validation"""
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(episode, step, is_best)
        
        # Check distributed coordination
        if not self.coordinator.coordinate_save("local", checkpoint_id):
            logger.info(f"Skipping checkpoint save due to coordination: {checkpoint_id}")
            return checkpoint_id
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'step': step,
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'model_config': self._get_model_config(model),
            'config': asdict(self.config)
        }
        
        # Add additional data
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Generate file path
        checkpoint_file = f"{self.config.checkpoint_prefix}_{checkpoint_id}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_file
        
        try:
            # Save checkpoint
            if self.config.async_saving:
                self._save_checkpoint_async(checkpoint_data, checkpoint_path, checkpoint_id, metrics)
            else:
                self._save_checkpoint_sync(checkpoint_data, checkpoint_path, checkpoint_id, metrics)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            raise
    
    def _save_checkpoint_sync(self, checkpoint_data: Dict[str, Any], 
                             checkpoint_path: Path, 
                             checkpoint_id: str, 
                             metrics: Dict[str, float]):
        """Save checkpoint synchronously"""
        # Save to file
        if self.config.use_compression:
            with gzip.open(checkpoint_path, 'wb') as f:
                torch.save(checkpoint_data, f)
        else:
            torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(checkpoint_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            version=self._get_next_version(),
            timestamp=datetime.now(),
            episode=checkpoint_data['episode'],
            step=checkpoint_data['step'],
            metrics=metrics,
            model_config=checkpoint_data['model_config'],
            file_path=str(checkpoint_path),
            file_size=checkpoint_path.stat().st_size,
            checksum=checksum
        )
        
        # Validate checkpoint
        if self.config.enable_validation:
            if not self.validator.validate_checkpoint(str(checkpoint_path), metadata):
                logger.error(f"Checkpoint validation failed: {checkpoint_id}")
                checkpoint_path.unlink()  # Remove invalid checkpoint
                return
            metadata.validated = True
        
        # Register checkpoint
        self.checkpoint_registry[checkpoint_id] = metadata
        self.coordinator.register_checkpoint(checkpoint_id, metadata)
        
        # Cloud backup
        if self.cloud_manager:
            if self.cloud_manager.upload_checkpoint(str(checkpoint_path), metadata):
                metadata.cloud_synced = True
        
        # Save registry
        self.save_checkpoint_registry()
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
    
    def _save_checkpoint_async(self, checkpoint_data: Dict[str, Any], 
                              checkpoint_path: Path, 
                              checkpoint_id: str, 
                              metrics: Dict[str, float]):
        """Save checkpoint asynchronously"""
        def save_task():
            self._save_checkpoint_sync(checkpoint_data, checkpoint_path, checkpoint_id, metrics)
        
        thread = threading.Thread(target=save_task)
        thread.daemon = True
        thread.start()
        self.background_tasks.append(thread)
    
    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load checkpoint with automatic recovery"""
        
        # Determine checkpoint to load
        if checkpoint_id is None:
            checkpoint_id = self._select_checkpoint_for_recovery()
        
        if checkpoint_id is None:
            logger.warning("No checkpoint found for recovery")
            return None
        
        # Get metadata
        metadata = self.checkpoint_registry.get(checkpoint_id)
        if metadata is None:
            logger.error(f"Checkpoint metadata not found: {checkpoint_id}")
            return None
        
        # Check local file
        checkpoint_path = metadata.file_path
        if not Path(checkpoint_path).exists():
            logger.warning(f"Local checkpoint not found: {checkpoint_path}")
            
            # Try cloud recovery
            if self.cloud_manager:
                logger.info(f"Attempting cloud recovery for: {checkpoint_id}")
                if self.cloud_manager.download_checkpoint(checkpoint_id, checkpoint_path):
                    logger.info(f"Checkpoint recovered from cloud: {checkpoint_id}")
                else:
                    logger.error(f"Cloud recovery failed: {checkpoint_id}")
                    return None
            else:
                logger.error(f"No cloud backup available for: {checkpoint_id}")
                return None
        
        # Validate checkpoint
        if self.config.enable_validation:
            if not self.validator.validate_checkpoint(checkpoint_path, metadata):
                logger.error(f"Checkpoint validation failed during load: {checkpoint_id}")
                return None
        
        # Load checkpoint
        try:
            if self.config.use_compression:
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f)
            else:
                checkpoint = torch.load(checkpoint_path)
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return None
    
    def _select_checkpoint_for_recovery(self) -> Optional[str]:
        """Select best checkpoint for recovery"""
        if not self.checkpoint_registry:
            return None
        
        if self.config.recovery_strategy == "latest":
            # Get latest checkpoint
            latest_metadata = max(
                self.checkpoint_registry.values(),
                key=lambda x: x.timestamp
            )
            return latest_metadata.checkpoint_id
            
        elif self.config.recovery_strategy == "best":
            # Get checkpoint with best metrics
            best_metadata = max(
                self.checkpoint_registry.values(),
                key=lambda x: x.metrics.get('reward', float('-inf'))
            )
            return best_metadata.checkpoint_id
            
        elif self.config.recovery_strategy == "stable":
            # Get most validated checkpoint
            stable_checkpoints = [
                metadata for metadata in self.checkpoint_registry.values()
                if metadata.validated and metadata.recovery_tested
            ]
            
            if stable_checkpoints:
                stable_metadata = max(stable_checkpoints, key=lambda x: x.timestamp)
                return stable_metadata.checkpoint_id
        
        # Fallback to latest
        latest_metadata = max(
            self.checkpoint_registry.values(),
            key=lambda x: x.timestamp
        )
        return latest_metadata.checkpoint_id
    
    def _generate_checkpoint_id(self, episode: int, step: int, is_best: bool) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self._get_next_version()
        
        suffix = "_best" if is_best else ""
        return f"ep{episode}_step{step}_{timestamp}_v{version}{suffix}"
    
    def _get_next_version(self) -> str:
        """Get next version string"""
        self.version_counter += 1
        
        if self.config.version_strategy == "semantic":
            major = self.version_counter // 1000
            minor = (self.version_counter % 1000) // 100
            patch = self.version_counter % 100
            return f"{major}.{minor}.{patch}"
        elif self.config.version_strategy == "timestamp":
            return datetime.now().strftime("%Y%m%d%H%M%S")
        else:  # hash
            return hashlib.md5(str(self.version_counter).encode()).hexdigest()[:8]
    
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration"""
        return {
            'model_class': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints"""
        if len(self.checkpoint_registry) <= self.config.max_checkpoints:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(
            self.checkpoint_registry.values(),
            key=lambda x: x.timestamp
        )
        
        # Keep max_checkpoints most recent
        to_remove = sorted_checkpoints[:-self.config.max_checkpoints]
        
        for metadata in to_remove:
            try:
                # Remove file
                Path(metadata.file_path).unlink(missing_ok=True)
                
                # Remove from registry
                del self.checkpoint_registry[metadata.checkpoint_id]
                
                logger.info(f"Removed old checkpoint: {metadata.checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {metadata.checkpoint_id}: {e}")
    
    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_task():
            while not self.shutdown_event.is_set():
                try:
                    self._health_check()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        thread = threading.Thread(target=monitoring_task, daemon=True)
        thread.start()
        self.background_tasks.append(thread)
    
    def _health_check(self):
        """Perform health check on checkpoints"""
        for checkpoint_id, metadata in self.checkpoint_registry.items():
            # Check file existence
            if not Path(metadata.file_path).exists():
                logger.warning(f"Checkpoint file missing: {checkpoint_id}")
                continue
            
            # Periodic validation
            if not metadata.validated:
                if self.validator.validate_checkpoint(metadata.file_path, metadata):
                    metadata.validated = True
                    logger.info(f"Checkpoint validated: {checkpoint_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
        self.cleanup()
    
    def save_checkpoint_registry(self):
        """Save checkpoint registry to disk"""
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        
        # Convert to serializable format
        registry_data = {
            checkpoint_id: metadata.to_dict()
            for checkpoint_id, metadata in self.checkpoint_registry.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def load_checkpoint_registry(self):
        """Load checkpoint registry from disk"""
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Convert back to metadata objects
            for checkpoint_id, data in registry_data.items():
                metadata = CheckpointMetadata.from_dict(data)
                self.checkpoint_registry[checkpoint_id] = metadata
            
            logger.info(f"Loaded {len(self.checkpoint_registry)} checkpoints from registry")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint registry: {e}")
    
    def load_version_counter(self):
        """Load version counter from disk"""
        version_file = self.checkpoint_dir / "version_counter.json"
        
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    self.version_counter = data.get('counter', 0)
            except Exception as e:
                logger.error(f"Failed to load version counter: {e}")
    
    def save_version_counter(self):
        """Save version counter to disk"""
        version_file = self.checkpoint_dir / "version_counter.json"
        
        with open(version_file, 'w') as f:
            json.dump({'counter': self.version_counter}, f)
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint information"""
        info = {
            'total_checkpoints': len(self.checkpoint_registry),
            'checkpoint_dir': str(self.checkpoint_dir),
            'config': asdict(self.config),
            'version_counter': self.version_counter,
            'checkpoints': {}
        }
        
        for checkpoint_id, metadata in self.checkpoint_registry.items():
            info['checkpoints'][checkpoint_id] = metadata.to_dict()
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up checkpoint manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for background tasks
        for task in self.background_tasks:
            if hasattr(task, 'join'):
                task.join(timeout=5)
        
        # Save state
        self.save_checkpoint_registry()
        self.save_version_counter()
        
        logger.info("Checkpoint manager cleanup completed")


def create_checkpoint_config(**kwargs) -> CheckpointConfig:
    """Create checkpoint configuration"""
    return CheckpointConfig(**kwargs)


def run_checkpoint_training_example():
    """Example of using advanced checkpoint manager"""
    
    # Create configuration
    config = create_checkpoint_config(
        checkpoint_dir="example_checkpoints",
        max_checkpoints=5,
        use_compression=True,
        enable_validation=True,
        enable_cloud_backup=False,
        async_saving=True
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = AdvancedCheckpointManager(config)
    
    # Create dummy model and optimizer
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with checkpointing
    for episode in range(50):
        # Simulate training
        loss = np.random.random()
        reward = np.random.random() * 100
        
        # Save checkpoint periodically
        if episode % 10 == 0:
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                episode=episode,
                step=episode * 100,
                metrics={'loss': loss, 'reward': reward},
                is_best=(episode == 40)  # Mark episode 40 as best
            )
            
            print(f"Saved checkpoint: {checkpoint_id}")
        
        # Simulate occasional recovery
        if episode == 25:
            print("Simulating recovery...")
            checkpoint = checkpoint_manager.load_checkpoint()
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Recovered from episode {checkpoint['episode']}")
    
    # Get final info
    info = checkpoint_manager.get_checkpoint_info()
    print(f"Final checkpoint info: {info['total_checkpoints']} checkpoints")
    
    # Cleanup
    checkpoint_manager.cleanup()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    run_checkpoint_training_example()