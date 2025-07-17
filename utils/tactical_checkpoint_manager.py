"""
Tactical Model Manager with Enhanced Metadata Tracking
Handles saving, loading, and management of tactical MARL models with comprehensive metadata
"""

import os
import torch
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class TacticalModelManager:
    """
    Enhanced model manager for tactical MARL system.
    
    Features:
    - Comprehensive metadata tracking alongside model weights
    - Best model selection based on multiple metrics
    - Automatic checkpoint rotation and cleanup
    - Performance trend analysis
    - Model comparison tools
    """
    
    def __init__(
        self,
        model_dir: str = "models/tactical",
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 10,
        best_metrics: List[str] = ["sharpe_ratio", "win_rate", "average_reward"],
        metadata_version: str = "1.0"
    ):
        """
        Initialize Tactical Model Manager.
        
        Args:
            model_dir: Directory to store models and metadata
            checkpoint_interval: Episodes between automatic checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            best_metrics: Metrics to track for best model selection
            metadata_version: Version of metadata format
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.best_metrics = best_metrics
        self.metadata_version = metadata_version
        
        # Subdirectories
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.metadata_dir = self.model_dir / "metadata"
        self.best_models_dir = self.model_dir / "best_models"
        
        # Create directories
        for dir_path in [self.checkpoints_dir, self.metadata_dir, self.best_models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Track best models
        self.best_models = {}
        for metric in self.best_metrics:
            self.best_models[metric] = {
                'value': float('-inf'),
                'checkpoint_path': None,
                'metadata_path': None
            }
        
        # Load existing best models
        self._load_best_models_registry()
        
        # Checkpoint history
        self.checkpoint_history = []
        self._load_checkpoint_history()
        
        logger.info(f"TacticalModelManager initialized at {self.model_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any],
        update_count: int,
        episode_count: int,
        metrics: Dict[str, float],
        buffer_state: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Tuple[str, str]:
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            model: Tactical MARL model
            optimizers: Dictionary of optimizers
            schedulers: Dictionary of schedulers
            update_count: Current update count
            episode_count: Current episode count
            metrics: Performance metrics
            buffer_state: Optional buffer state
            additional_info: Additional information to save
            is_best: Whether this is manually marked as best
            
        Returns:
            Tuple of (checkpoint_path, metadata_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint filename
        checkpoint_name = f"tactical_marl_ep{episode_count}_upd{update_count}_{timestamp}.pt"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        # Create metadata filename
        metadata_name = f"tactical_marl_ep{episode_count}_upd{update_count}_{timestamp}.json"
        metadata_path = self.metadata_dir / metadata_name
        
        # Prepare comprehensive metadata
        metadata = self._create_metadata(
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            update_count=update_count,
            episode_count=episode_count,
            metrics=metrics,
            buffer_state=buffer_state,
            additional_info=additional_info,
            checkpoint_path=str(checkpoint_path),
            timestamp=timestamp
        )
        
        # Save metadata first
        self._save_metadata(metadata, metadata_path)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizers_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
            'schedulers_state_dict': {name: sched.state_dict() if hasattr(sched, 'state_dict') else None 
                                     for name, sched in schedulers.items()},
            'update_count': update_count,
            'episode_count': episode_count,
            'timestamp': timestamp,
            'metadata_path': str(metadata_path),
            'buffer_state': buffer_state,
            'additional_info': additional_info or {}
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        self._update_checkpoint_history(checkpoint_path, metadata_path, metadata)
        
        # Check if this is a new best model
        self._check_and_update_best_models(checkpoint_path, metadata_path, metrics, is_best)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_name}")
        logger.info(f"Metadata saved: {metadata_name}")
        
        return str(checkpoint_path), str(metadata_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any],
        load_optimizers: bool = True,
        load_schedulers: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint and return metadata.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizers: Optimizers to load state into
            schedulers: Schedulers to load state into
            load_optimizers: Whether to load optimizer states
            load_schedulers: Whether to load scheduler states
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer states
        if load_optimizers and 'optimizers_state_dict' in checkpoint_data:
            for name, optimizer in optimizers.items():
                if name in checkpoint_data['optimizers_state_dict']:
                    optimizer.load_state_dict(checkpoint_data['optimizers_state_dict'][name])
        
        # Load scheduler states
        if load_schedulers and 'schedulers_state_dict' in checkpoint_data:
            for name, scheduler in schedulers.items():
                if name in checkpoint_data['schedulers_state_dict']:
                    scheduler_state = checkpoint_data['schedulers_state_dict'][name]
                    if scheduler_state is not None and hasattr(scheduler, 'load_state_dict'):
                        scheduler.load_state_dict(scheduler_state)
        
        # Load metadata if available
        metadata = None
        if 'metadata_path' in checkpoint_data:
            metadata_path = checkpoint_data['metadata_path']
            if os.path.exists(metadata_path):
                metadata = self._load_metadata(metadata_path)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return {
            'checkpoint_data': checkpoint_data,
            'metadata': metadata,
            'update_count': checkpoint_data.get('update_count', 0),
            'episode_count': checkpoint_data.get('episode_count', 0),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'buffer_state': checkpoint_data.get('buffer_state'),
            'additional_info': checkpoint_data.get('additional_info', {})
        }
    
    def get_best_checkpoint(self, metric: str = "sharpe_ratio") -> Optional[Tuple[str, str]]:
        """
        Get path to best checkpoint for specified metric.
        
        Args:
            metric: Metric to use for best model selection
            
        Returns:
            Tuple of (checkpoint_path, metadata_path) or None if not found
        """
        if metric not in self.best_models:
            logger.warning(f"Metric {metric} not tracked")
            return None
        
        best_model = self.best_models[metric]
        if best_model['checkpoint_path'] is None:
            logger.warning(f"No best model found for metric {metric}")
            return None
        
        return best_model['checkpoint_path'], best_model['metadata_path']
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get history of all checkpoints."""
        return self.checkpoint_history.copy()
    
    def get_performance_trends(self, metric: str, window: int = 10) -> Dict[str, Any]:
        """
        Get performance trends for specified metric.
        
        Args:
            metric: Metric to analyze
            window: Window size for trend analysis
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.checkpoint_history) < 2:
            return {'trend': 'insufficient_data', 'values': []}
        
        # Extract metric values
        values = []
        episodes = []
        
        for checkpoint_info in self.checkpoint_history:
            if metric in checkpoint_info.get('metrics', {}):
                values.append(checkpoint_info['metrics'][metric])
                episodes.append(checkpoint_info['episode_count'])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'values': values}
        
        # Calculate trend
        recent_values = values[-window:] if len(values) >= window else values
        
        # Simple trend analysis
        if len(recent_values) >= 2:
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            if trend_slope > 0.001:
                trend = 'improving'
            elif trend_slope < -0.001:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'values': values,
            'episodes': episodes,
            'recent_values': recent_values,
            'slope': trend_slope if 'trend_slope' in locals() else 0,
            'best_value': max(values),
            'worst_value': min(values),
            'current_value': values[-1] if values else 0,
            'improvement_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        }
    
    def compare_models(self, checkpoint_paths: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models based on their metadata.
        
        Args:
            checkpoint_paths: List of checkpoint paths to compare
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'models': [],
            'best_by_metric': {},
            'summary': {}
        }
        
        # Load metadata for each model
        for checkpoint_path in checkpoint_paths:
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                metadata_path = checkpoint_data.get('metadata_path')
                
                if metadata_path and os.path.exists(metadata_path):
                    metadata = self._load_metadata(metadata_path)
                    comparison['models'].append({
                        'checkpoint_path': checkpoint_path,
                        'metadata': metadata,
                        'episode_count': checkpoint_data.get('episode_count', 0),
                        'update_count': checkpoint_data.get('update_count', 0)
                    })
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        # Find best model for each metric
        for metric in self.best_metrics:
            best_model = None
            best_value = float('-inf')
            
            for model_info in comparison['models']:
                metrics = model_info['metadata'].get('performance_metrics', {})
                if metric in metrics:
                    value = metrics[metric]
                    if value > best_value:
                        best_value = value
                        best_model = model_info['checkpoint_path']
            
            comparison['best_by_metric'][metric] = {
                'checkpoint_path': best_model,
                'value': best_value
            }
        
        # Summary statistics
        if comparison['models']:
            comparison['summary'] = {
                'total_models': len(comparison['models']),
                'episode_range': [
                    min(m['episode_count'] for m in comparison['models']),
                    max(m['episode_count'] for m in comparison['models'])
                ],
                'update_range': [
                    min(m['update_count'] for m in comparison['models']),
                    max(m['update_count'] for m in comparison['models'])
                ]
            }
        
        return comparison
    
    def _create_metadata(
        self,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any],
        update_count: int,
        episode_count: int,
        metrics: Dict[str, float],
        buffer_state: Optional[Dict[str, Any]],
        additional_info: Optional[Dict[str, Any]],
        checkpoint_path: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """Create comprehensive metadata dictionary."""
        # Get model info
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        
        # Create metadata
        metadata = {
            'metadata_version': self.metadata_version,
            'timestamp': timestamp,
            'checkpoint_path': checkpoint_path,
            
            # Training state
            'training_state': {
                'update_count': update_count,
                'episode_count': episode_count,
                'timestamp': timestamp
            },
            
            # Model architecture
            'model_architecture': {
                'model_type': 'TacticalMARLSystem',
                'total_parameters': model_info.get('total_parameters', 0),
                'trainable_parameters': model_info.get('trainable_parameters', 0),
                'input_shape': model_info.get('input_shape', (60, 7)),
                'action_dim': model_info.get('action_dim', 3),
                'agent_parameters': model_info.get('agent_parameters', {}),
                'critic_parameters': model_info.get('critic_parameters', 0)
            },
            
            # Performance metrics
            'performance_metrics': metrics.copy(),
            
            # Optimizer information
            'optimizers': {},
            
            # Scheduler information
            'schedulers': {},
            
            # Buffer state summary
            'buffer_state': {},
            
            # Additional information
            'additional_info': additional_info or {}
        }
        
        # Add optimizer info
        for name, optimizer in optimizers.items():
            metadata['optimizers'][name] = {
                'type': type(optimizer).__name__,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'parameter_groups': len(optimizer.param_groups)
            }
        
        # Add scheduler info
        for name, scheduler in schedulers.items():
            metadata['schedulers'][name] = {
                'type': type(scheduler).__name__,
                'last_lr': getattr(scheduler, 'get_last_lr', lambda: [0])()[0] if hasattr(scheduler, 'get_last_lr') else 0
            }
        
        # Add buffer state summary
        if buffer_state:
            metadata['buffer_state'] = {
                'capacity': buffer_state.get('capacity', 0),
                'size': buffer_state.get('tree_n_entries', 0),
                'utilization': buffer_state.get('tree_n_entries', 0) / max(buffer_state.get('capacity', 1), 1),
                'total_samples': buffer_state.get('total_samples', 0),
                'beta': buffer_state.get('beta', 0),
                'alpha': buffer_state.get('alpha', 0)
            }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any], metadata_path: Path):
        """Save metadata to JSON file."""
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _update_checkpoint_history(self, checkpoint_path: Path, metadata_path: Path, metadata: Dict[str, Any]):
        """Update checkpoint history."""
        history_entry = {
            'checkpoint_path': str(checkpoint_path),
            'metadata_path': str(metadata_path),
            'timestamp': metadata['timestamp'],
            'episode_count': metadata['training_state']['episode_count'],
            'update_count': metadata['training_state']['update_count'],
            'metrics': metadata['performance_metrics']
        }
        
        self.checkpoint_history.append(history_entry)
        
        # Save history
        history_path = self.model_dir / 'checkpoint_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def _check_and_update_best_models(self, checkpoint_path: Path, metadata_path: Path, metrics: Dict[str, float], is_best: bool):
        """Check if this is a new best model and update registry."""
        updated_metrics = []
        
        for metric in self.best_metrics:
            if metric in metrics:
                current_value = metrics[metric]
                best_value = self.best_models[metric]['value']
                
                if current_value > best_value or is_best:
                    # Update best model
                    self.best_models[metric] = {
                        'value': current_value,
                        'checkpoint_path': str(checkpoint_path),
                        'metadata_path': str(metadata_path)
                    }
                    updated_metrics.append(metric)
                    
                    # Copy to best models directory
                    best_checkpoint_name = f"best_{metric}_tactical_marl.pt"
                    best_metadata_name = f"best_{metric}_tactical_marl.json"
                    
                    best_checkpoint_path = self.best_models_dir / best_checkpoint_name
                    best_metadata_path = self.best_models_dir / best_metadata_name
                    
                    shutil.copy2(checkpoint_path, best_checkpoint_path)
                    shutil.copy2(metadata_path, best_metadata_path)
        
        if updated_metrics:
            logger.info(f"New best model for metrics: {updated_metrics}")
            self._save_best_models_registry()
    
    def _save_best_models_registry(self):
        """Save best models registry."""
        registry_path = self.model_dir / 'best_models_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.best_models, f, indent=2)
    
    def _load_best_models_registry(self):
        """Load best models registry."""
        registry_path = self.model_dir / 'best_models_registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                loaded_registry = json.load(f)
                self.best_models.update(loaded_registry)
    
    def _load_checkpoint_history(self):
        """Load checkpoint history."""
        history_path = self.model_dir / 'checkpoint_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.checkpoint_history = json.load(f)
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints beyond max_checkpoints."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by episode count
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['episode_count'])
        
        # Keep only the most recent checkpoints
        to_remove = sorted_history[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            try:
                # Remove checkpoint file
                checkpoint_path = Path(checkpoint_info['checkpoint_path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove metadata file
                metadata_path = Path(checkpoint_info['metadata_path'])
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from history
                self.checkpoint_history.remove(checkpoint_info)
                
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_info['checkpoint_path']}: {e}")
        
        # Save updated history
        history_path = self.model_dir / 'checkpoint_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model management summary."""
        return {
            'model_dir': str(self.model_dir),
            'checkpoint_interval': self.checkpoint_interval,
            'max_checkpoints': self.max_checkpoints,
            'best_metrics': self.best_metrics,
            'total_checkpoints': len(self.checkpoint_history),
            'best_models': self.best_models,
            'latest_checkpoint': self.checkpoint_history[-1] if self.checkpoint_history else None,
            'directory_structure': {
                'checkpoints': len(list(self.checkpoints_dir.glob('*.pt'))),
                'metadata': len(list(self.metadata_dir.glob('*.json'))),
                'best_models': len(list(self.best_models_dir.glob('*.pt')))
            }
        }