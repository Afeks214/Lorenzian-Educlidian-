#!/usr/bin/env python3
"""
Comprehensive Backup and Checkpoint System for Training Infrastructure
Handles model checkpoints, data backups, and system state management
"""

import os
import shutil
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import pickle
import threading
from contextlib import contextmanager
import gzip
import tarfile
from enum import Enum

class BackupType(Enum):
    """Types of backups"""
    CHECKPOINT = "checkpoint"
    FULL_BACKUP = "full_backup"
    INCREMENTAL = "incremental"
    EMERGENCY = "emergency"

@dataclass
class BackupConfig:
    """Backup system configuration"""
    backup_dir: str = "/home/QuantNova/GrandModel/colab/infrastructure/backup"
    max_checkpoints: int = 10
    max_backups: int = 5
    backup_interval_hours: int = 24
    checkpoint_interval_steps: int = 1000
    compress_backups: bool = True
    verify_backups: bool = True
    backup_retention_days: int = 7
    enable_auto_backup: bool = True
    backup_formats: List[str] = None

    def __post_init__(self):
        if self.backup_formats is None:
            self.backup_formats = ['torch', 'pickle', 'json']

@dataclass
class CheckpointMetadata:
    """Checkpoint metadata"""
    timestamp: float
    epoch: int
    step: int
    loss: float
    model_state_size: int
    optimizer_state_size: int
    file_path: str
    file_size: int
    checksum: str
    training_time: float
    validation_metrics: Optional[Dict[str, float]] = None

@dataclass
class BackupMetadata:
    """Backup metadata"""
    timestamp: float
    backup_type: BackupType
    file_path: str
    file_size: int
    checksum: str
    items_backed_up: List[str]
    backup_duration: float
    compressed: bool

class BackupSystem:
    """Comprehensive backup and checkpoint system"""
    
    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.backup_dir = Path(self.config.backup_dir)
        self.checkpoint_dir = self.backup_dir / "checkpoints"
        self.model_dir = self.backup_dir / "models"
        self.config_dir = self.backup_dir / "configs"
        self.log_dir = self.backup_dir / "logs"
        
        for dir_path in [self.backup_dir, self.checkpoint_dir, self.model_dir, self.config_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.checkpoint_metadata: List[CheckpointMetadata] = []
        self.backup_metadata: List[BackupMetadata] = []
        
        # Auto-backup thread
        self.auto_backup_active = False
        self.auto_backup_thread = None
        
        # Load existing metadata
        self._load_metadata()
        
        # Start auto-backup if enabled
        if self.config.enable_auto_backup:
            self.start_auto_backup()
        
        self.logger.info(f"Backup system initialized with directory: {self.backup_dir}")
    
    def _load_metadata(self):
        """Load existing metadata"""
        try:
            checkpoint_metadata_file = self.backup_dir / "checkpoint_metadata.json"
            if checkpoint_metadata_file.exists():
                with open(checkpoint_metadata_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_metadata = [
                        CheckpointMetadata(**item) for item in data
                    ]
            
            backup_metadata_file = self.backup_dir / "backup_metadata.json"
            if backup_metadata_file.exists():
                with open(backup_metadata_file, 'r') as f:
                    data = json.load(f)
                    self.backup_metadata = [
                        BackupMetadata(**{**item, 'backup_type': BackupType(item['backup_type'])})
                        for item in data
                    ]
        
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            # Save checkpoint metadata
            checkpoint_metadata_file = self.backup_dir / "checkpoint_metadata.json"
            with open(checkpoint_metadata_file, 'w') as f:
                json.dump([asdict(item) for item in self.checkpoint_metadata], f, indent=2)
            
            # Save backup metadata
            backup_metadata_file = self.backup_dir / "backup_metadata.json"
            with open(backup_metadata_file, 'w') as f:
                data = []
                for item in self.backup_metadata:
                    item_dict = asdict(item)
                    item_dict['backup_type'] = item.backup_type.value
                    data.append(item_dict)
                json.dump(data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, step: int, loss: float, 
                         validation_metrics: Optional[Dict[str, float]] = None,
                         training_time: float = 0.0) -> str:
        """Create a training checkpoint"""
        timestamp = time.time()
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}_step_{step:05d}_{int(timestamp)}"
        
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': timestamp,
            'validation_metrics': validation_metrics or {},
            'training_time': training_time
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate sizes and checksum
        file_size = checkpoint_path.stat().st_size
        checksum = self._calculate_checksum(checkpoint_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            loss=loss,
            model_state_size=len(pickle.dumps(model.state_dict())),
            optimizer_state_size=len(pickle.dumps(optimizer.state_dict())),
            file_path=str(checkpoint_path),
            file_size=file_size,
            checksum=checksum,
            training_time=training_time,
            validation_metrics=validation_metrics
        )
        
        self.checkpoint_metadata.append(metadata)
        self._save_metadata()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Checkpoint created: {checkpoint_path} (size: {file_size/1024/1024:.2f}MB)")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint"""
        if checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoint_metadata:
                raise ValueError("No checkpoints available")
            
            latest_checkpoint = max(self.checkpoint_metadata, key=lambda x: x.timestamp)
            checkpoint_path = latest_checkpoint.file_path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Verify checksum
        if self.config.verify_backups:
            expected_checksum = None
            for metadata in self.checkpoint_metadata:
                if metadata.file_path == str(checkpoint_path):
                    expected_checksum = metadata.checksum
                    break
            
            if expected_checksum:
                actual_checksum = self._calculate_checksum(checkpoint_path)
                if actual_checksum != expected_checksum:
                    raise ValueError(f"Checkpoint checksum mismatch: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
    
    def restore_from_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                              checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Restore model and optimizer from checkpoint"""
        checkpoint_data = self.load_checkpoint(checkpoint_path)
        
        # Restore model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Return training state
        return {
            'epoch': checkpoint_data['epoch'],
            'step': checkpoint_data['step'],
            'loss': checkpoint_data['loss'],
            'validation_metrics': checkpoint_data.get('validation_metrics', {}),
            'training_time': checkpoint_data.get('training_time', 0.0)
        }
    
    def create_backup(self, items_to_backup: List[str], 
                     backup_type: BackupType = BackupType.FULL_BACKUP,
                     backup_name: Optional[str] = None) -> str:
        """Create a backup of specified items"""
        start_time = time.time()
        timestamp = int(start_time)
        
        if backup_name is None:
            backup_name = f"backup_{backup_type.value}_{timestamp}"
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # Create backup
        with tarfile.open(backup_path, "w:gz" if self.config.compress_backups else "w") as tar:
            for item in items_to_backup:
                item_path = Path(item)
                if item_path.exists():
                    tar.add(item_path, arcname=item_path.name)
                else:
                    self.logger.warning(f"Item not found for backup: {item}")
        
        backup_duration = time.time() - start_time
        file_size = backup_path.stat().st_size
        checksum = self._calculate_checksum(backup_path)
        
        # Create metadata
        metadata = BackupMetadata(
            timestamp=start_time,
            backup_type=backup_type,
            file_path=str(backup_path),
            file_size=file_size,
            checksum=checksum,
            items_backed_up=items_to_backup,
            backup_duration=backup_duration,
            compressed=self.config.compress_backups
        )
        
        self.backup_metadata.append(metadata)
        self._save_metadata()
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        self.logger.info(f"Backup created: {backup_path} (size: {file_size/1024/1024:.2f}MB, "
                        f"duration: {backup_duration:.2f}s)")
        return str(backup_path)
    
    def restore_backup(self, backup_path: str, restore_dir: str):
        """Restore from backup"""
        backup_path = Path(backup_path)
        restore_dir = Path(restore_dir)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Verify checksum
        if self.config.verify_backups:
            expected_checksum = None
            for metadata in self.backup_metadata:
                if metadata.file_path == str(backup_path):
                    expected_checksum = metadata.checksum
                    break
            
            if expected_checksum:
                actual_checksum = self._calculate_checksum(backup_path)
                if actual_checksum != expected_checksum:
                    raise ValueError(f"Backup checksum mismatch: {backup_path}")
        
        # Extract backup
        restore_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(backup_path, "r:gz" if backup_path.suffix == ".gz" else "r") as tar:
            tar.extractall(restore_dir)
        
        self.logger.info(f"Backup restored to: {restore_dir}")
    
    def emergency_backup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, step: int, loss: float) -> str:
        """Create emergency backup"""
        timestamp = int(time.time())
        emergency_name = f"emergency_backup_{timestamp}"
        
        # Create emergency checkpoint
        checkpoint_path = self.create_checkpoint(
            model, optimizer, epoch, step, loss, training_time=0.0
        )
        
        # Create emergency backup with essential items
        essential_items = [
            checkpoint_path,
            str(self.backup_dir / "checkpoint_metadata.json"),
            str(self.backup_dir / "backup_metadata.json")
        ]
        
        backup_path = self.create_backup(
            essential_items, 
            BackupType.EMERGENCY, 
            emergency_name
        )
        
        self.logger.warning(f"Emergency backup created: {backup_path}")
        return backup_path
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints"""
        if len(self.checkpoint_metadata) <= self.config.max_checkpoints:
            return
        
        # Sort by timestamp and keep newest
        sorted_checkpoints = sorted(self.checkpoint_metadata, key=lambda x: x.timestamp, reverse=True)
        checkpoints_to_remove = sorted_checkpoints[self.config.max_checkpoints:]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint.file_path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update metadata
        self.checkpoint_metadata = sorted_checkpoints[:self.config.max_checkpoints]
        self._save_metadata()
    
    def _cleanup_old_backups(self):
        """Clean up old backups"""
        if len(self.backup_metadata) <= self.config.max_backups:
            return
        
        # Sort by timestamp and keep newest
        sorted_backups = sorted(self.backup_metadata, key=lambda x: x.timestamp, reverse=True)
        backups_to_remove = sorted_backups[self.config.max_backups:]
        
        for backup in backups_to_remove:
            backup_path = Path(backup.file_path)
            if backup_path.exists():
                backup_path.unlink()
                self.logger.info(f"Removed old backup: {backup_path}")
        
        # Update metadata
        self.backup_metadata = sorted_backups[:self.config.max_backups]
        self._save_metadata()
    
    def start_auto_backup(self):
        """Start automatic backup thread"""
        if self.auto_backup_active:
            return
        
        self.auto_backup_active = True
        self.auto_backup_thread = threading.Thread(target=self._auto_backup_loop)
        self.auto_backup_thread.daemon = True
        self.auto_backup_thread.start()
        
        self.logger.info("Auto-backup started")
    
    def stop_auto_backup(self):
        """Stop automatic backup thread"""
        self.auto_backup_active = False
        if self.auto_backup_thread:
            self.auto_backup_thread.join()
        
        self.logger.info("Auto-backup stopped")
    
    def _auto_backup_loop(self):
        """Auto-backup loop"""
        while self.auto_backup_active:
            try:
                # Sleep for backup interval
                time.sleep(self.config.backup_interval_hours * 3600)
                
                if not self.auto_backup_active:
                    break
                
                # Create automatic backup
                backup_items = [
                    str(self.checkpoint_dir),
                    str(self.backup_dir / "checkpoint_metadata.json"),
                    str(self.backup_dir / "backup_metadata.json")
                ]
                
                self.create_backup(backup_items, BackupType.INCREMENTAL)
                
            except Exception as e:
                self.logger.error(f"Auto-backup error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status"""
        return {
            'checkpoint_count': len(self.checkpoint_metadata),
            'backup_count': len(self.backup_metadata),
            'latest_checkpoint': max(self.checkpoint_metadata, key=lambda x: x.timestamp).timestamp if self.checkpoint_metadata else None,
            'latest_backup': max(self.backup_metadata, key=lambda x: x.timestamp).timestamp if self.backup_metadata else None,
            'total_checkpoint_size': sum(c.file_size for c in self.checkpoint_metadata),
            'total_backup_size': sum(b.file_size for b in self.backup_metadata),
            'auto_backup_active': self.auto_backup_active,
            'backup_dir': str(self.backup_dir)
        }
    
    def save_backup_report(self, filepath: str):
        """Save backup system report"""
        report = {
            'config': asdict(self.config),
            'status': self.get_backup_status(),
            'checkpoint_metadata': [asdict(c) for c in self.checkpoint_metadata],
            'backup_metadata': [
                {**asdict(b), 'backup_type': b.backup_type.value} 
                for b in self.backup_metadata
            ],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Backup report saved to {filepath}")
    
    @contextmanager
    def backup_context(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      epoch: int, step: int, loss: float):
        """Context manager for backup protection"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in backup context: {e}")
            # Create emergency backup
            self.emergency_backup(model, optimizer, epoch, step, loss)
            raise
    
    def cleanup(self):
        """Cleanup backup system"""
        self.stop_auto_backup()
        self._save_metadata()
        
        self.logger.info("Backup system cleaned up")

# Factory functions
def create_backup_system(backup_dir: str = None, 
                        max_checkpoints: int = 10,
                        enable_auto_backup: bool = True) -> BackupSystem:
    """Create backup system with default settings"""
    config = BackupConfig(
        backup_dir=backup_dir or "/home/QuantNova/GrandModel/colab/infrastructure/backup",
        max_checkpoints=max_checkpoints,
        enable_auto_backup=enable_auto_backup
    )
    
    return BackupSystem(config)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backup system
    backup_system = create_backup_system()
    
    # Example model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint
    checkpoint_path = backup_system.create_checkpoint(
        model, optimizer, epoch=1, step=100, loss=0.5
    )
    
    # Create backup
    backup_path = backup_system.create_backup([checkpoint_path])
    
    # Get status
    status = backup_system.get_backup_status()
    print(f"Backup Status: {json.dumps(status, indent=2)}")
    
    # Save report
    backup_system.save_backup_report("/tmp/backup_report.json")
    
    # Cleanup
    backup_system.cleanup()