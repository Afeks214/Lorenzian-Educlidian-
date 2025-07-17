"""
Checkpoint Manager for Strategic MARL System
Handles saving and loading of model states, optimizers, and training metrics
"""

import os
import torch
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpointing and recovery for MAPPO training.
    
    Features:
    - Save complete training state (models, optimizers, schedulers, metrics)
    - Automatic checkpoint rotation with configurable retention
    - Best model tracking based on specified metric
    - Training state recovery for seamless resumption
    - Metadata tracking for experiment reproducibility
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        best_metric: str = "reward",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of recent checkpoints to keep
            best_metric: Metric name to track for best model
            mode: "max" if higher metric is better, "min" otherwise
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.mode = mode
        
        # Track best metric value
        self.best_metric_value = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None
        
        # Checkpoint history
        self.checkpoint_history = []
        self._load_checkpoint_history()
        
    def save_checkpoint(
        self,
        agents: Dict[str, torch.nn.Module],
        critic: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Optional[Dict[str, Any]] = None,
        update_count: int = 0,
        episode_count: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        buffer_state: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete training checkpoint.
        
        Args:
            agents: Dictionary of agent models
            critic: Centralized critic model
            optimizers: Dictionary of optimizers (one per agent + critic)
            schedulers: Optional learning rate schedulers
            update_count: Current update iteration
            episode_count: Current episode count
            metrics: Current training metrics
            buffer_state: Optional replay buffer state
            additional_info: Any additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_update_{update_count}_ep_{episode_count}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'update_count': update_count,
            'episode_count': episode_count,
            'timestamp': timestamp,
            'metrics': metrics or {},
            
            # Model states
            'agent_states': {
                name: agent.state_dict() for name, agent in agents.items()
            },
            'critic_state': critic.state_dict(),
            
            # Optimizer states
            'optimizer_states': {
                name: opt.state_dict() for name, opt in optimizers.items()
            },
            
            # Optional components
            'scheduler_states': {
                name: sched.state_dict() for name, sched in (schedulers or {}).items()
            },
            'buffer_state': buffer_state,
            'additional_info': additional_info or {}
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'update_count': update_count,
            'episode_count': episode_count,
            'timestamp': timestamp,
            'metrics': metrics or {}
        })
        
        # Check if this is the best checkpoint
        if metrics and self.best_metric in metrics:
            metric_value = metrics[self.best_metric]
            is_best = (
                (self.mode == 'max' and metric_value > self.best_metric_value) or
                (self.mode == 'min' and metric_value < self.best_metric_value)
            )
            
            if is_best:
                self.best_metric_value = metric_value
                self._save_best_checkpoint(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save checkpoint history
        self._save_checkpoint_history()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        agents: Optional[Dict[str, torch.nn.Module]] = None,
        critic: Optional[torch.nn.Module] = None,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Dict[str, Any]] = None,
        load_best: bool = False,
        map_location: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to specific checkpoint (if None, loads latest)
            agents: Agent models to load states into
            critic: Critic model to load state into
            optimizers: Optimizers to load states into
            schedulers: Schedulers to load states into
            load_best: If True, load best checkpoint instead of latest
            map_location: Device to map tensors to
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            if load_best and self.best_checkpoint_path:
                checkpoint_path = self.best_checkpoint_path
            else:
                checkpoint_path = self._get_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return {}
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Restore model states
        if agents:
            for name, agent in agents.items():
                if name in checkpoint['agent_states']:
                    agent.load_state_dict(checkpoint['agent_states'][name])
                    logger.info(f"Loaded state for agent: {name}")
        
        if critic and 'critic_state' in checkpoint:
            critic.load_state_dict(checkpoint['critic_state'])
            logger.info("Loaded critic state")
        
        # Restore optimizer states
        if optimizers:
            for name, opt in optimizers.items():
                if name in checkpoint['optimizer_states']:
                    opt.load_state_dict(checkpoint['optimizer_states'][name])
                    logger.info(f"Loaded optimizer state: {name}")
        
        # Restore scheduler states
        if schedulers and 'scheduler_states' in checkpoint:
            for name, sched in schedulers.items():
                if name in checkpoint['scheduler_states']:
                    sched.load_state_dict(checkpoint['scheduler_states'][name])
                    logger.info(f"Loaded scheduler state: {name}")
        
        return checkpoint
    
    def save_model_only(
        self,
        agents: Dict[str, torch.nn.Module],
        critic: torch.nn.Module,
        save_path: Optional[str] = None
    ) -> str:
        """
        Save only model weights (no training state).
        
        Args:
            agents: Agent models
            critic: Critic model
            save_path: Optional custom save path
            
        Returns:
            Path to saved models
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.checkpoint_dir / f"models_only_{timestamp}.pt"
        
        model_dict = {
            'agents': {name: agent.state_dict() for name, agent in agents.items()},
            'critic': critic.state_dict()
        }
        
        torch.save(model_dict, save_path)
        logger.info(f"Saved model weights to {save_path}")
        
        return str(save_path)
    
    def load_model_only(
        self,
        load_path: str,
        agents: Dict[str, torch.nn.Module],
        critic: torch.nn.Module,
        map_location: str = 'cpu'
    ):
        """
        Load only model weights.
        
        Args:
            load_path: Path to model file
            agents: Agent models to load into
            critic: Critic model to load into
            map_location: Device to map tensors to
        """
        model_dict = torch.load(load_path, map_location=map_location)
        
        for name, agent in agents.items():
            if name in model_dict['agents']:
                agent.load_state_dict(model_dict['agents'][name])
                
        critic.load_state_dict(model_dict['critic'])
        logger.info(f"Loaded model weights from {load_path}")
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all saved checkpoints."""
        return self.checkpoint_history
    
    def _save_best_checkpoint(self, checkpoint_path: Path):
        """Save a copy as the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        shutil.copy2(checkpoint_path, best_path)
        self.best_checkpoint_path = str(best_path)
        
        # Save best checkpoint metadata
        metadata = {
            'original_path': str(checkpoint_path),
            'best_metric': self.best_metric,
            'best_value': self.best_metric_value,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_dir / "best_checkpoint_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved best checkpoint with {self.best_metric}={self.best_metric_value}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by update count
            sorted_history = sorted(
                self.checkpoint_history, 
                key=lambda x: x['update_count']
            )
            
            # Remove oldest checkpoints
            to_remove = sorted_history[:-self.max_checkpoints]
            
            for checkpoint_info in to_remove:
                checkpoint_path = checkpoint_info['path']
                if os.path.exists(checkpoint_path) and checkpoint_path != self.best_checkpoint_path:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                    
            # Update history
            self.checkpoint_history = sorted_history[-self.max_checkpoints:]
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        if not self.checkpoint_history:
            return None
            
        latest = max(self.checkpoint_history, key=lambda x: x['update_count'])
        return latest['path'] if os.path.exists(latest['path']) else None
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.checkpoint_history = json.load(f)
                
        # Load best checkpoint info
        best_info_path = self.checkpoint_dir / "best_checkpoint_info.json"
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                info = json.load(f)
                self.best_metric_value = info['best_value']
                self.best_checkpoint_path = str(self.checkpoint_dir / "best_checkpoint.pt")


class ModelVersionManager:
    """
    Manages different versions of trained models for A/B testing and rollback.
    """
    
    def __init__(self, base_dir: str = "model_versions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.version_info = self._load_version_info()
        
    def save_version(
        self,
        agents: Dict[str, torch.nn.Module],
        critic: torch.nn.Module,
        version_name: str,
        metrics: Dict[str, float],
        description: str = ""
    ) -> str:
        """Save a new model version."""
        version_dir = self.base_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # Save models
        model_path = version_dir / "models.pt"
        torch.save({
            'agents': {name: agent.state_dict() for name, agent in agents.items()},
            'critic': critic.state_dict()
        }, model_path)
        
        # Save version info
        version_data = {
            'name': version_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'description': description,
            'model_path': str(model_path)
        }
        
        with open(version_dir / "version_info.json", 'w') as f:
            json.dump(version_data, f, indent=2)
            
        # Update global version registry
        self.version_info[version_name] = version_data
        self._save_version_info()
        
        logger.info(f"Saved model version: {version_name}")
        return str(model_path)
    
    def load_version(
        self,
        version_name: str,
        agents: Dict[str, torch.nn.Module],
        critic: torch.nn.Module,
        map_location: str = 'cpu'
    ):
        """Load a specific model version."""
        if version_name not in self.version_info:
            raise ValueError(f"Version {version_name} not found")
            
        model_path = self.version_info[version_name]['model_path']
        model_dict = torch.load(model_path, map_location=map_location)
        
        for name, agent in agents.items():
            if name in model_dict['agents']:
                agent.load_state_dict(model_dict['agents'][name])
                
        critic.load_state_dict(model_dict['critic'])
        logger.info(f"Loaded model version: {version_name}")
    
    def list_versions(self) -> Dict[str, Dict[str, Any]]:
        """List all available model versions."""
        return self.version_info
    
    def _save_version_info(self):
        """Save version registry."""
        with open(self.base_dir / "version_registry.json", 'w') as f:
            json.dump(self.version_info, f, indent=2)
    
    def _load_version_info(self) -> Dict[str, Dict[str, Any]]:
        """Load version registry."""
        registry_path = self.base_dir / "version_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {}