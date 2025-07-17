"""
Advanced Learning Rate Scheduling for MARL Training
Implements sophisticated learning rate scheduling strategies for optimal convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers"""
    # Base settings
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Warmup settings
    warmup_steps: int = 1000
    warmup_strategy: str = "linear"  # "linear", "cosine", "exponential"
    
    # Adaptive settings
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 0
    
    # Cyclical settings
    cycle_length: int = 2000
    cycle_mult: float = 1.0
    
    # Cosine annealing
    T_max: int = 1000
    eta_min: float = 0.0
    
    # OneCycle settings
    max_lr_ratio: float = 10.0
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    pct_start: float = 0.3
    
    # Plateau settings
    mode: str = "min"  # "min", "max"
    verbose: bool = False
    
    # Custom settings
    milestones: List[int] = None
    gamma: float = 0.1
    
    # Multi-step settings
    step_size: int = 30
    
    # Exponential decay
    decay_rate: float = 0.95
    decay_steps: int = 100
    
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = [1000, 2000, 3000]


class BaseScheduler(ABC):
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        self.lr_history = []
        self.metric_history = []
        self.last_epoch = -1
        
    @abstractmethod
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups"""
        pass
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Update learning rates"""
        self.step_count += 1
        
        # Store metrics
        if metrics:
            self.metric_history.append(metrics)
        
        # Get new learning rates
        new_lrs = self.get_lr()
        
        # Update optimizer
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        
        # Store history
        self.lr_history.append(new_lrs[0])  # Store first group's LR
        
        self.last_epoch += 1
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state"""
        return {
            'step_count': self.step_count,
            'lr_history': self.lr_history,
            'metric_history': self.metric_history,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state"""
        self.step_count = state_dict['step_count']
        self.lr_history = state_dict['lr_history']
        self.metric_history = state_dict['metric_history']
        self.last_epoch = state_dict['last_epoch']


class WarmupScheduler(BaseScheduler):
    """Learning rate warmup scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig, base_scheduler: Optional[BaseScheduler] = None):
        super().__init__(optimizer, config)
        self.base_scheduler = base_scheduler
        self.warmup_complete = False
        
    def get_lr(self) -> List[float]:
        """Get learning rates with warmup"""
        if self.step_count < self.config.warmup_steps:
            # Warmup phase
            return self._get_warmup_lr()
        else:
            # Post-warmup phase
            if not self.warmup_complete:
                self.warmup_complete = True
                logger.info(f"Warmup completed after {self.config.warmup_steps} steps")
            
            if self.base_scheduler:
                return self.base_scheduler.get_lr()
            else:
                return [self.config.initial_lr] * len(self.optimizer.param_groups)
    
    def _get_warmup_lr(self) -> List[float]:
        """Calculate warmup learning rates"""
        progress = self.step_count / self.config.warmup_steps
        
        if self.config.warmup_strategy == "linear":
            factor = progress
        elif self.config.warmup_strategy == "cosine":
            factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.config.warmup_strategy == "exponential":
            factor = math.exp(-5 * (1 - progress))
        else:
            factor = progress
        
        return [self.config.initial_lr * factor] * len(self.optimizer.param_groups)


class CosineAnnealingWithRestartsScheduler(BaseScheduler):
    """Cosine annealing with warm restarts"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        super().__init__(optimizer, config)
        self.T_cur = 0
        self.T_i = config.cycle_length
        self.T_mult = config.cycle_mult
        self.restart_count = 0
        
    def get_lr(self) -> List[float]:
        """Get learning rates using cosine annealing with restarts"""
        if self.T_cur >= self.T_i:
            # Restart
            self.restart_count += 1
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
            logger.info(f"Cosine annealing restart #{self.restart_count}, next cycle length: {self.T_i}")
        
        # Cosine annealing
        lr = self.config.min_lr + (self.config.max_lr - self.config.min_lr) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        self.T_cur += 1
        
        return [lr] * len(self.optimizer.param_groups)


class OneCycleScheduler(BaseScheduler):
    """One cycle learning rate scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig, total_steps: int):
        super().__init__(optimizer, config)
        self.total_steps = total_steps
        self.max_lr = config.initial_lr * config.max_lr_ratio
        self.min_lr = config.initial_lr / config.div_factor
        self.final_lr = config.initial_lr / config.final_div_factor
        self.step_size_up = int(config.pct_start * total_steps)
        self.step_size_down = total_steps - self.step_size_up
        
    def get_lr(self) -> List[float]:
        """Get learning rates using one cycle policy"""
        if self.step_count <= self.step_size_up:
            # Ascending phase
            pct = self.step_count / self.step_size_up
            lr = self.min_lr + (self.max_lr - self.min_lr) * pct
        else:
            # Descending phase
            pct = (self.step_count - self.step_size_up) / self.step_size_down
            lr = self.max_lr - (self.max_lr - self.final_lr) * pct
        
        return [lr] * len(self.optimizer.param_groups)


class AdaptiveLRScheduler(BaseScheduler):
    """Adaptive learning rate scheduler based on metrics"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        super().__init__(optimizer, config)
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.in_cooldown = False
        
    def get_lr(self) -> List[float]:
        """Get learning rates based on metric improvement"""
        if not self.metric_history:
            return [self.config.initial_lr] * len(self.optimizer.param_groups)
        
        current_metric = self._get_current_metric()
        
        if self.best_metric is None:
            self.best_metric = current_metric
        
        # Check if metric improved
        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.in_cooldown = False
            self.cooldown_counter = 0
        else:
            self.num_bad_epochs += 1
        
        # Check cooldown
        if self.in_cooldown:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.config.cooldown:
                self.in_cooldown = False
                self.cooldown_counter = 0
        
        # Reduce learning rate if needed
        current_lrs = self.get_last_lr()
        if self.num_bad_epochs >= self.config.patience and not self.in_cooldown:
            new_lrs = [lr * self.config.factor for lr in current_lrs]
            new_lrs = [max(lr, self.config.min_lr) for lr in new_lrs]
            
            if any(new_lr < current_lr for new_lr, current_lr in zip(new_lrs, current_lrs)):
                logger.info(f"Reducing learning rate by factor {self.config.factor}")
                self.num_bad_epochs = 0
                self.in_cooldown = True
                self.cooldown_counter = 0
            
            return new_lrs
        
        return current_lrs
    
    def _get_current_metric(self) -> float:
        """Get current metric value"""
        if not self.metric_history:
            return 0.0
        
        latest_metrics = self.metric_history[-1]
        return latest_metrics.get('loss', 0.0)
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best"""
        if self.config.mode == "min":
            return current < best - self.config.threshold
        else:
            return current > best + self.config.threshold


class CyclicalLRScheduler(BaseScheduler):
    """Cyclical learning rate scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        super().__init__(optimizer, config)
        self.cycle_length = config.cycle_length
        self.base_lr = config.min_lr
        self.max_lr = config.max_lr
        
    def get_lr(self) -> List[float]:
        """Get learning rates using cyclical policy"""
        cycle_position = self.step_count % self.cycle_length
        
        if cycle_position < self.cycle_length // 2:
            # Ascending phase
            pct = cycle_position / (self.cycle_length // 2)
            lr = self.base_lr + (self.max_lr - self.base_lr) * pct
        else:
            # Descending phase
            pct = (cycle_position - self.cycle_length // 2) / (self.cycle_length // 2)
            lr = self.max_lr - (self.max_lr - self.base_lr) * pct
        
        return [lr] * len(self.optimizer.param_groups)


class ExponentialDecayScheduler(BaseScheduler):
    """Exponential decay scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        super().__init__(optimizer, config)
        
    def get_lr(self) -> List[float]:
        """Get learning rates using exponential decay"""
        decay_factor = math.pow(self.config.decay_rate, self.step_count / self.config.decay_steps)
        lr = self.config.initial_lr * decay_factor
        lr = max(lr, self.config.min_lr)
        
        return [lr] * len(self.optimizer.param_groups)


class PolyDecayScheduler(BaseScheduler):
    """Polynomial decay scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig, total_steps: int, power: float = 0.9):
        super().__init__(optimizer, config)
        self.total_steps = total_steps
        self.power = power
        
    def get_lr(self) -> List[float]:
        """Get learning rates using polynomial decay"""
        if self.step_count >= self.total_steps:
            return [self.config.min_lr] * len(self.optimizer.param_groups)
        
        decay_factor = (1 - self.step_count / self.total_steps) ** self.power
        lr = self.config.initial_lr * decay_factor
        lr = max(lr, self.config.min_lr)
        
        return [lr] * len(self.optimizer.param_groups)


class NoamScheduler(BaseScheduler):
    """Noam scheduler (used in Transformers)"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig, model_size: int):
        super().__init__(optimizer, config)
        self.model_size = model_size
        
    def get_lr(self) -> List[float]:
        """Get learning rates using Noam formula"""
        arg1 = self.step_count ** (-0.5)
        arg2 = self.step_count * (self.config.warmup_steps ** (-1.5))
        
        lr = (self.model_size ** (-0.5)) * min(arg1, arg2)
        
        return [lr] * len(self.optimizer.param_groups)


class LookAheadScheduler(BaseScheduler):
    """Look-ahead learning rate scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig, look_ahead_steps: int = 5):
        super().__init__(optimizer, config)
        self.look_ahead_steps = look_ahead_steps
        self.future_metrics = deque(maxlen=look_ahead_steps)
        
    def get_lr(self) -> List[float]:
        """Get learning rates using look-ahead strategy"""
        current_lrs = self.get_last_lr() if self.lr_history else [self.config.initial_lr]
        
        # If we have enough history, use look-ahead
        if len(self.metric_history) >= self.look_ahead_steps:
            recent_metrics = self.metric_history[-self.look_ahead_steps:]
            
            # Calculate trend
            losses = [m.get('loss', 0) for m in recent_metrics]
            trend = np.polyfit(range(len(losses)), losses, 1)[0]
            
            # Adjust learning rate based on trend
            if trend > 0:  # Loss increasing
                factor = 0.95
            elif trend < -0.01:  # Loss decreasing significantly
                factor = 1.05
            else:  # Stable
                factor = 1.0
            
            new_lrs = [lr * factor for lr in current_lrs]
            new_lrs = [max(min(lr, self.config.max_lr), self.config.min_lr) for lr in new_lrs]
            
            return new_lrs
        
        return current_lrs


class AdvancedLRSchedulerManager:
    """Manager for advanced learning rate scheduling"""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        self.optimizer = optimizer
        self.config = config
        self.schedulers = {}
        self.active_scheduler = None
        self.scheduler_history = []
        
        # Performance tracking
        self.performance_history = []
        self.lr_performance_map = defaultdict(list)
        
        # Adaptive scheduler switching
        self.switch_threshold = 0.01
        self.switch_patience = 20
        self.no_improvement_count = 0
        
        logger.info(f"Advanced LR scheduler manager initialized")
    
    def add_scheduler(self, name: str, scheduler: BaseScheduler):
        """Add a scheduler to the manager"""
        self.schedulers[name] = scheduler
        logger.info(f"Added scheduler: {name}")
    
    def set_active_scheduler(self, name: str):
        """Set active scheduler"""
        if name not in self.schedulers:
            raise ValueError(f"Scheduler {name} not found")
        
        self.active_scheduler = self.schedulers[name]
        self.scheduler_history.append({
            'step': len(self.performance_history),
            'scheduler': name,
            'timestamp': time.time()
        })
        
        logger.info(f"Activated scheduler: {name}")
    
    def step(self, metrics: Dict[str, float]):
        """Step the active scheduler"""
        if self.active_scheduler is None:
            logger.warning("No active scheduler set")
            return
        
        # Store performance
        self.performance_history.append(metrics)
        
        # Track LR performance
        current_lr = self.active_scheduler.get_last_lr()[0]
        self.lr_performance_map[current_lr].append(metrics.get('loss', 0))
        
        # Step scheduler
        self.active_scheduler.step(metrics)
        
        # Check if we should switch schedulers
        if self._should_switch_scheduler():
            self._switch_to_best_scheduler()
    
    def _should_switch_scheduler(self) -> bool:
        """Check if we should switch schedulers"""
        if len(self.performance_history) < self.switch_patience:
            return False
        
        # Check recent performance
        recent_performance = self.performance_history[-self.switch_patience:]
        recent_losses = [p.get('loss', 0) for p in recent_performance]
        
        # Check if performance is stagnating
        if len(recent_losses) >= 2:
            improvement = recent_losses[0] - recent_losses[-1]
            if improvement < self.switch_threshold:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
        
        return self.no_improvement_count >= self.switch_patience
    
    def _switch_to_best_scheduler(self):
        """Switch to the best performing scheduler"""
        if len(self.schedulers) <= 1:
            return
        
        # Evaluate all schedulers
        scheduler_performance = {}
        
        for name, scheduler in self.schedulers.items():
            if name == self.get_active_scheduler_name():
                continue
            
            # Create a copy and simulate performance
            perf_score = self._evaluate_scheduler(scheduler)
            scheduler_performance[name] = perf_score
        
        # Find best scheduler
        if scheduler_performance:
            best_scheduler = max(scheduler_performance, key=scheduler_performance.get)
            logger.info(f"Switching to best scheduler: {best_scheduler}")
            self.set_active_scheduler(best_scheduler)
            self.no_improvement_count = 0
    
    def _evaluate_scheduler(self, scheduler: BaseScheduler) -> float:
        """Evaluate scheduler performance"""
        # Simple evaluation based on expected convergence
        # This is a placeholder - in practice, you'd use more sophisticated evaluation
        return np.random.random()
    
    def get_active_scheduler_name(self) -> str:
        """Get name of active scheduler"""
        for name, scheduler in self.schedulers.items():
            if scheduler == self.active_scheduler:
                return name
        return "unknown"
    
    def get_lr_history(self) -> List[float]:
        """Get learning rate history"""
        if self.active_scheduler:
            return self.active_scheduler.lr_history
        return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {
            'active_scheduler': self.get_active_scheduler_name(),
            'total_steps': len(self.performance_history),
            'scheduler_switches': len(self.scheduler_history),
            'lr_history': self.get_lr_history(),
            'performance_history': self.performance_history,
            'scheduler_history': self.scheduler_history
        }
        
        # Add LR performance analysis
        if self.lr_performance_map:
            lr_analysis = {}
            for lr, performances in self.lr_performance_map.items():
                lr_analysis[lr] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'count': len(performances)
                }
            report['lr_analysis'] = lr_analysis
        
        return report
    
    def visualize_performance(self, save_path: str = None):
        """Visualize scheduler performance"""
        if not self.performance_history:
            logger.warning("No performance history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Learning rate history
        steps = range(len(self.get_lr_history()))
        axes[0, 0].plot(steps, self.get_lr_history(), 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].set_title('Learning Rate History')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. Loss history
        losses = [p.get('loss', 0) for p in self.performance_history]
        axes[0, 1].plot(losses, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss History')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. LR vs Performance
        if self.lr_performance_map:
            lrs = list(self.lr_performance_map.keys())
            mean_perfs = [np.mean(self.lr_performance_map[lr]) for lr in lrs]
            axes[1, 0].scatter(lrs, mean_perfs, alpha=0.7)
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Mean Loss')
            axes[1, 0].set_title('LR vs Performance')
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scheduler switches
        if self.scheduler_history:
            switch_steps = [sh['step'] for sh in self.scheduler_history]
            scheduler_names = [sh['scheduler'] for sh in self.scheduler_history]
            
            for i, (step, name) in enumerate(zip(switch_steps, scheduler_names)):
                axes[1, 1].axvline(x=step, color=f'C{i}', linestyle='--', alpha=0.7, label=name)
            
            axes[1, 1].plot(losses, 'k-', alpha=0.5)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Scheduler Switches')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def create_scheduler_config(**kwargs) -> SchedulerConfig:
    """Create scheduler configuration"""
    return SchedulerConfig(**kwargs)


def create_scheduler_suite(optimizer: optim.Optimizer, config: SchedulerConfig, total_steps: int = 10000) -> AdvancedLRSchedulerManager:
    """Create a suite of schedulers"""
    manager = AdvancedLRSchedulerManager(optimizer, config)
    
    # Add different schedulers
    manager.add_scheduler("warmup", WarmupScheduler(optimizer, config))
    manager.add_scheduler("cosine_restart", CosineAnnealingWithRestartsScheduler(optimizer, config))
    manager.add_scheduler("one_cycle", OneCycleScheduler(optimizer, config, total_steps))
    manager.add_scheduler("adaptive", AdaptiveLRScheduler(optimizer, config))
    manager.add_scheduler("cyclical", CyclicalLRScheduler(optimizer, config))
    manager.add_scheduler("exponential", ExponentialDecayScheduler(optimizer, config))
    manager.add_scheduler("poly", PolyDecayScheduler(optimizer, config, total_steps))
    manager.add_scheduler("look_ahead", LookAheadScheduler(optimizer, config))
    
    # Set default active scheduler
    manager.set_active_scheduler("warmup")
    
    return manager


def run_scheduler_example():
    """Example of using advanced learning rate scheduling"""
    
    # Create model and optimizer
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create scheduler configuration
    config = create_scheduler_config(
        initial_lr=0.001,
        min_lr=1e-6,
        max_lr=0.01,
        warmup_steps=100,
        patience=10,
        factor=0.5
    )
    
    # Create scheduler suite
    scheduler_manager = create_scheduler_suite(optimizer, config, total_steps=1000)
    
    # Simulate training
    for step in range(1000):
        # Simulate loss
        loss = 1.0 * np.exp(-step / 200) + 0.1 * np.random.random()
        
        # Create metrics
        metrics = {
            'loss': loss,
            'accuracy': 1.0 - loss,
            'step': step
        }
        
        # Step scheduler
        scheduler_manager.step(metrics)
        
        # Log progress
        if step % 100 == 0:
            current_lr = scheduler_manager.active_scheduler.get_last_lr()[0]
            active_scheduler = scheduler_manager.get_active_scheduler_name()
            print(f"Step {step}: LR={current_lr:.6f}, Loss={loss:.4f}, Scheduler={active_scheduler}")
    
    # Get performance report
    report = scheduler_manager.get_performance_report()
    print(f"\nTraining completed!")
    print(f"Total scheduler switches: {report['scheduler_switches']}")
    print(f"Final scheduler: {report['active_scheduler']}")
    
    # Visualize results
    scheduler_manager.visualize_performance("scheduler_performance.png")
    
    return scheduler_manager


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    scheduler_manager = run_scheduler_example()
    
    print("Example completed successfully!")