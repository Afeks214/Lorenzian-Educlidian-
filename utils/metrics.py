"""
Training Metrics Tracking and Visualization
Provides comprehensive metric tracking, logging, and analysis utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Comprehensive metrics tracking for MAPPO training.
    
    Features:
    - Real-time metric tracking
    - Moving averages and statistics
    - Performance analysis
    - Metric visualization
    - CSV/JSON export
    """
    
    def __init__(
        self,
        window_size: int = 100,
        save_dir: Optional[str] = None,
        log_interval: int = 10
    ):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving averages
            save_dir: Directory to save metric logs
            log_interval: Interval for logging metrics
        """
        self.window_size = window_size
        self.log_interval = log_interval
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.windowed_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_metrics = defaultdict(list)
        
        # Performance tracking
        self.best_metrics = {}
        self.metric_stats = defaultdict(lambda: {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0,
            'count': 0
        })
        
        # Timing
        self.timers = defaultdict(list)
        self.start_time = time.time()
        
        # Save directory
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
            
        # Update counter
        self.update_count = 0
        
    def add_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Add a scalar metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step count
        """
        if step is None:
            step = self.update_count
            
        self.metrics[name].append((step, value))
        self.windowed_metrics[name].append(value)
        
        # Update statistics
        stats = self.metric_stats[name]
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['sum'] += value
        stats['count'] += 1
        
        # Track best metrics
        if 'loss' in name.lower():
            if name not in self.best_metrics or value < self.best_metrics[name][1]:
                self.best_metrics[name] = (step, value)
        else:
            if name not in self.best_metrics or value > self.best_metrics[name][1]:
                self.best_metrics[name] = (step, value)
                
    def add_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Add multiple scalar metrics at once."""
        for name, value in metrics.items():
            self.add_scalar(name, value, step)
            
    def add_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Add histogram data.
        
        Args:
            name: Metric name
            values: Array of values
            step: Optional step count
        """
        if step is None:
            step = self.update_count
            
        histogram_data = {
            'step': step,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75))
        }
        
        if name not in self.metrics:
            self.metrics[f'{name}_histogram'] = []
        self.metrics[f'{name}_histogram'].append(histogram_data)
        
    def add_episode_metrics(self, metrics: Dict[str, float], episode: int):
        """Add metrics for a complete episode."""
        for name, value in metrics.items():
            self.episode_metrics[name].append((episode, value))
            
    def start_timer(self, name: str):
        """Start a timer for performance tracking."""
        self.timers[f'{name}_start'] = time.time()
        
    def end_timer(self, name: str):
        """End a timer and record duration."""
        if f'{name}_start' in self.timers:
            duration = time.time() - self.timers[f'{name}_start']
            self.add_scalar(f'timing/{name}', duration)
            del self.timers[f'{name}_start']
            
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values (latest or windowed average)."""
        current = {}
        
        for name, values in self.windowed_metrics.items():
            if values:
                current[name] = np.mean(values)
                
        return current
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        if metric_name not in self.metric_stats:
            return {}
            
        stats = self.metric_stats[metric_name]
        return {
            'min': stats['min'],
            'max': stats['max'],
            'mean': stats['sum'] / max(stats['count'], 1),
            'count': stats['count']
        }
        
    def should_log(self) -> bool:
        """Check if metrics should be logged."""
        return self.update_count % self.log_interval == 0
        
    def log_metrics(self, prefix: str = ""):
        """Log current metrics to console."""
        if not self.should_log():
            return
            
        metrics = self.get_current_metrics()
        
        # Format metrics for logging
        log_str = f"{prefix} Update {self.update_count} | "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        logger.info(log_str)
        
    def save_metrics(self):
        """Save metrics to disk."""
        if self.save_dir is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.save_dir / f"metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'metrics': {k: list(v) for k, v in self.metrics.items()},
                'statistics': dict(self.metric_stats),
                'best_metrics': self.best_metrics,
                'config': {
                    'window_size': self.window_size,
                    'total_updates': self.update_count,
                    'duration': time.time() - self.start_time
                }
            }, f, indent=2)
            
        # Save as CSV for easy analysis
        csv_path = self.save_dir / f"metrics_{timestamp}.csv"
        
        # Convert to DataFrame
        data = []
        for metric_name, values in self.metrics.items():
            if '_histogram' not in metric_name:
                for step, value in values:
                    data.append({
                        'step': step,
                        'metric': metric_name,
                        'value': value
                    })
                    
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved metrics to {json_path} and {csv_path}")
        
    def plot_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot training metrics.
        
        Args:
            metric_names: List of metrics to plot (None for all)
            save_path: Path to save plot
            show: Whether to display plot
        """
        if metric_names is None:
            metric_names = [k for k in self.metrics.keys() if '_histogram' not in k]
            
        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            if metric_name not in self.metrics:
                continue
                
            ax = axes[i] if n_metrics > 1 else axes[0]
            
            # Get data
            data = self.metrics[metric_name]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            
            # Plot line
            ax.plot(steps, values, alpha=0.7, label=metric_name)
            
            # Add moving average
            if len(values) > self.window_size:
                ma = pd.Series(values).rolling(self.window_size).mean()
                ax.plot(steps, ma, linewidth=2, label=f'{metric_name} (MA)')
                
            # Formatting
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value')
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Remove unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive summary report."""
        report = {
            'training_duration': time.time() - self.start_time,
            'total_updates': self.update_count,
            'metrics_summary': {},
            'best_metrics': self.best_metrics,
            'final_metrics': self.get_current_metrics()
        }
        
        # Add metric summaries
        for metric_name in self.metrics:
            if '_histogram' not in metric_name:
                stats = self.get_statistics(metric_name)
                report['metrics_summary'][metric_name] = stats
                
        return report
        
    def update(self):
        """Increment update counter."""
        self.update_count += 1


class EpisodeTracker:
    """
    Tracks episode-level metrics and statistics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_count = 0
        
        # Episode storage
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_episode(
        self,
        total_reward: float,
        episode_length: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Add episode data."""
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.episode_metrics[key].append(value)
                
    def get_statistics(self) -> Dict[str, float]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
            
        stats = {
            'episode_count': self.episode_count,
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'success_rate': np.mean([r > 0 for r in self.episode_rewards])
        }
        
        # Add custom metrics
        for key, values in self.episode_metrics.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                
        return stats


class PerformanceMonitor:
    """
    Monitors training performance and efficiency.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = deque(maxlen=1000)
        self.update_times = deque(maxlen=100)
        self.last_step_time = time.time()
        
    def record_step(self):
        """Record a single environment step."""
        current_time = time.time()
        self.step_times.append(current_time - self.last_step_time)
        self.last_step_time = current_time
        
    def record_update(self, duration: float):
        """Record an update duration."""
        self.update_times.append(duration)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        current_time = time.time()
        total_duration = current_time - self.start_time
        
        metrics = {
            'total_duration_hours': total_duration / 3600,
            'avg_step_time_ms': np.mean(self.step_times) * 1000 if self.step_times else 0,
            'avg_update_time_s': np.mean(self.update_times) if self.update_times else 0,
            'steps_per_second': 1 / np.mean(self.step_times) if self.step_times else 0
        }
        
        return metrics