"""
Learning Rate Schedulers for MAPPO Training
Implements various scheduling strategies for adaptive learning
"""

import numpy as np
from typing import Optional, Callable
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearDecayScheduler(_LRScheduler):
    """
    Linear learning rate decay scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_lr: float,
        end_lr: float,
        decay_steps: int,
        last_epoch: int = -1
    ):
        """
        Initialize linear decay scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            start_lr: Starting learning rate
            end_lr: Final learning rate
            decay_steps: Number of steps for decay
            last_epoch: Last epoch number
        """
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        
        # Set initial lr
        for group in optimizer.param_groups:
            group['lr'] = start_lr
            
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch >= self.decay_steps:
            return [self.end_lr for _ in self.base_lrs]
        
        progress = self.last_epoch / self.decay_steps
        current_lr = self.start_lr + (self.end_lr - self.start_lr) * progress
        
        return [current_lr for _ in self.base_lrs]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm restarts.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cosine annealing with warm restarts.
        
        Args:
            optimizer: Optimizer to schedule
            T_0: Number of iterations for the first restart
            T_mult: Factor to increase T_i after restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch number
            verbose: Whether to print lr updates
        """
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_i:
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs
            ]
        else:
            return self.get_lr()
            
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        np.log(
                            epoch / self.T_0 * (self.T_mult - 1) + 1
                        ) / np.log(self.T_mult)
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AdaptiveScheduler:
    """
    Adaptive learning rate scheduler based on performance metrics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'max',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
        cooldown: int = 0
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: 'max' if higher metric is better, 'min' otherwise
            factor: Factor to reduce lr by
            patience: Number of epochs with no improvement
            threshold: Threshold for measuring improvement
            min_lr: Minimum learning rate
            cooldown: Epochs to wait after reduction
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.cooldown = cooldown
        
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        
        if mode == 'max':
            self.mode_worse = -np.inf
            self.is_better = lambda a, b: a > b + b * threshold
        else:
            self.mode_worse = np.inf
            self.is_better = lambda a, b: a < b - b * threshold
            
    def step(self, metric: float):
        """
        Step scheduler based on metric.
        
        Args:
            metric: Performance metric to track
        """
        current = metric
        
        if self.best is None:
            self.best = current
            
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
            
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class CyclicScheduler:
    """
    Cyclic learning rate scheduler for exploring lr ranges.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle'
    ):
        """
        Initialize cyclic scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            base_lr: Lower learning rate boundary
            max_lr: Upper learning rate boundary
            step_size_up: Number of steps in increasing phase
            step_size_down: Number of steps in decreasing phase
            mode: One of 'triangular', 'triangular2', 'exp_range'
            gamma: Constant for 'exp_range' mode
            scale_fn: Custom scaling function
            scale_mode: 'cycle' or 'iterations'
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        self.clr_iterations = 0
        self.cycle_iterations = 0
        self.cycle_count = 0
        
        self._set_learning_rate()
        
    def _set_learning_rate(self):
        """Set the learning rate."""
        cycle = 1 + self.clr_iterations // (self.step_size_up + self.step_size_down)
        x = self.clr_iterations / (self.step_size_up + self.step_size_down)
        
        if x <= 0.5:
            scale_factor = 2 * x
        else:
            scale_factor = 2 * (1 - x)
            
        if self.scale_fn:
            scale_factor = self.scale_fn(self.cycle_iterations if self.scale_mode == 'iterations' else cycle)
        elif self.mode == 'triangular':
            scale_factor = scale_factor
        elif self.mode == 'triangular2':
            scale_factor = scale_factor / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = scale_factor * (self.gamma ** self.clr_iterations)
            
        lr = self.base_lr + (self.max_lr - self.base_lr) * scale_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def step(self):
        """Step the scheduler."""
        self.clr_iterations += 1
        self.cycle_iterations += 1
        
        if self.cycle_iterations >= self.step_size_up + self.step_size_down:
            self.cycle_iterations = 0
            self.cycle_count += 1
            
        self._set_learning_rate()
        
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class SchedulerManager:
    """
    Manages multiple schedulers for different components.
    """
    
    def __init__(self):
        self.schedulers = {}
        
    def add_scheduler(self, name: str, scheduler):
        """Add a scheduler."""
        self.schedulers[name] = scheduler
        
    def step_all(self, metric: Optional[float] = None):
        """Step all schedulers."""
        for name, scheduler in self.schedulers.items():
            if isinstance(scheduler, AdaptiveScheduler) and metric is not None:
                scheduler.step(metric)
            else:
                scheduler.step()
                
    def get_lrs(self) -> dict:
        """Get current learning rates."""
        lrs = {}
        for name, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'get_last_lr'):
                lrs[name] = scheduler.get_last_lr()[0]
            else:
                lrs[name] = scheduler.optimizer.param_groups[0]['lr']
        return lrs
        
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            name: scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {}
            for name, scheduler in self.schedulers.items()
        }
        
    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        for name, scheduler in self.schedulers.items():
            if name in state_dict and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(state_dict[name])