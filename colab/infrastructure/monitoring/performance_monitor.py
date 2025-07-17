"""
Advanced Performance Monitoring System for MARL Training Infrastructure
Real-time monitoring of training performance, resource usage, and system health
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque, defaultdict
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Thread, Lock
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for MARL training
    Tracks GPU, CPU, memory, training metrics, and system health
    """
    
    def __init__(self, buffer_size: int = 10000, log_dir: str = "logs"):
        self.buffer_size = buffer_size
        self.log_dir = log_dir
        self.start_time = time.time()
        self.lock = Lock()
        
        # Create directories
        os.makedirs(f"{log_dir}/performance", exist_ok=True)
        os.makedirs(f"{log_dir}/system", exist_ok=True)
        
        # Initialize monitoring buffers
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.system_buffer = deque(maxlen=buffer_size)
        self.training_buffer = deque(maxlen=buffer_size)
        self.gpu_buffer = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.alerts = []
        self.thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 95.0,
            'gpu_memory': 95.0,
            'gpu_utilization': 95.0,
            'training_time_per_episode': 300.0,  # 5 minutes
            'loss_divergence_threshold': 100.0
        }
        
        # System info
        self.system_info = self._get_system_info()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"Performance Monitor initialized with device: {self.device}")
        logger.info(f"System: {self.system_info}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': psutil.uname().system,
            'python_version': psutil.uname().release,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_version': torch.version.cuda
            })
        
        return system_info
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitor_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(f"Background monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Background monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = time.time()
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_metrics = {}
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            gpu_metrics = {
                'gpu_memory_used_gb': gpu_memory_used / (1024**3),
                'gpu_memory_total_gb': gpu_memory_total / (1024**3),
                'gpu_memory_percent': gpu_memory_percent,
                'gpu_memory_available_gb': (gpu_memory_total - gpu_memory_used) / (1024**3)
            }
        
        # System metrics
        system_metrics = {
            'timestamp': timestamp,
            'elapsed_time': timestamp - self.start_time,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            **gpu_metrics
        }
        
        # Add to buffer
        with self.lock:
            self.system_buffer.append(system_metrics)
        
        # Check thresholds and create alerts
        self._check_thresholds(system_metrics)
        
        # Log critical metrics
        if system_metrics['cpu_percent'] > 80 or system_metrics['memory_percent'] > 80:
            logger.warning(f"High resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check if metrics exceed thresholds and create alerts"""
        alerts = []
        
        if metrics['cpu_percent'] > self.thresholds['cpu_usage']:
            alerts.append(f"HIGH CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > self.thresholds['memory_usage']:
            alerts.append(f"HIGH Memory usage: {metrics['memory_percent']:.1f}%")
        
        if 'gpu_memory_percent' in metrics and metrics['gpu_memory_percent'] > self.thresholds['gpu_memory']:
            alerts.append(f"HIGH GPU memory: {metrics['gpu_memory_percent']:.1f}%")
        
        if alerts:
            with self.lock:
                self.alerts.extend(alerts)
            logger.warning(f"Performance alerts: {alerts}")
    
    def log_training_step(self, episode: int, step: int, metrics: Dict[str, float]):
        """Log training step metrics"""
        timestamp = time.time()
        
        training_data = {
            'timestamp': timestamp,
            'episode': episode,
            'step': step,
            'elapsed_time': timestamp - self.start_time,
            **metrics
        }
        
        with self.lock:
            self.training_buffer.append(training_data)
        
        # Check for training anomalies
        self._check_training_anomalies(training_data)
    
    def _check_training_anomalies(self, training_data: Dict[str, Any]):
        """Check for training anomalies and performance issues"""
        # Check for loss divergence
        if 'total_loss' in training_data:
            if training_data['total_loss'] > self.thresholds['loss_divergence_threshold']:
                logger.warning(f"Loss divergence detected: {training_data['total_loss']:.4f}")
        
        # Check for training time anomalies
        if len(self.training_buffer) > 1:
            prev_data = list(self.training_buffer)[-2]
            time_diff = training_data['timestamp'] - prev_data['timestamp']
            if time_diff > self.thresholds['training_time_per_episode']:
                logger.warning(f"Slow training detected: {time_diff:.2f}s per step")
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model-specific performance metrics"""
        timestamp = time.time()
        
        model_data = {
            'timestamp': timestamp,
            'model_name': model_name,
            'elapsed_time': timestamp - self.start_time,
            **metrics
        }
        
        with self.lock:
            self.metrics_buffer.append(model_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            current_time = time.time()
            
            # System metrics summary
            system_data = list(self.system_buffer)
            training_data = list(self.training_buffer)
            
            summary = {
                'monitoring_duration_hours': (current_time - self.start_time) / 3600,
                'total_samples': len(system_data),
                'alerts_count': len(self.alerts),
                'recent_alerts': self.alerts[-10:] if self.alerts else [],
                'system_info': self.system_info
            }
            
            if system_data:
                latest_system = system_data[-1]
                cpu_usage = [s['cpu_percent'] for s in system_data[-100:]]
                memory_usage = [s['memory_percent'] for s in system_data[-100:]]
                
                summary['current_status'] = {
                    'cpu_percent': latest_system['cpu_percent'],
                    'memory_percent': latest_system['memory_percent'],
                    'gpu_memory_percent': latest_system.get('gpu_memory_percent', 0),
                    'uptime_hours': latest_system['elapsed_time'] / 3600
                }
                
                summary['performance_stats'] = {
                    'avg_cpu_usage': np.mean(cpu_usage),
                    'max_cpu_usage': np.max(cpu_usage),
                    'avg_memory_usage': np.mean(memory_usage),
                    'max_memory_usage': np.max(memory_usage),
                    'cpu_usage_std': np.std(cpu_usage),
                    'memory_usage_std': np.std(memory_usage)
                }
            
            if training_data:
                episodes = [t['episode'] for t in training_data]
                summary['training_progress'] = {
                    'current_episode': max(episodes) if episodes else 0,
                    'total_training_steps': len(training_data),
                    'episodes_completed': len(set(episodes))
                }
                
                # Calculate training speed
                if len(training_data) > 1:
                    time_per_step = np.mean([
                        training_data[i]['timestamp'] - training_data[i-1]['timestamp']
                        for i in range(1, len(training_data))
                    ])
                    summary['training_speed'] = {
                        'avg_time_per_step': time_per_step,
                        'steps_per_minute': 60 / time_per_step if time_per_step > 0 else 0
                    }
            
            return summary
    
    def save_performance_report(self, filepath: str = None):
        """Save comprehensive performance report"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.log_dir}/performance/performance_report_{timestamp}.json"
        
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {filepath}")
        return filepath
    
    def plot_performance_metrics(self, save_path: str = None):
        """Create comprehensive performance plots"""
        if not self.system_buffer:
            logger.warning("No data to plot")
            return
        
        # Prepare data
        system_data = list(self.system_buffer)
        df_system = pd.DataFrame(system_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Infrastructure Performance Monitor', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(df_system['elapsed_time'] / 3600, df_system['cpu_percent'])
        axes[0, 0].axhline(y=self.thresholds['cpu_usage'], color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_xlabel('Hours')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Memory Usage
        axes[0, 1].plot(df_system['elapsed_time'] / 3600, df_system['memory_percent'])
        axes[0, 1].axhline(y=self.thresholds['memory_usage'], color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Hours')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # GPU Memory (if available)
        if 'gpu_memory_percent' in df_system.columns:
            axes[1, 0].plot(df_system['elapsed_time'] / 3600, df_system['gpu_memory_percent'])
            axes[1, 0].axhline(y=self.thresholds['gpu_memory'], color='red', linestyle='--', label='Threshold')
            axes[1, 0].set_title('GPU Memory Usage Over Time')
            axes[1, 0].set_xlabel('Hours')
            axes[1, 0].set_ylabel('GPU Memory Usage (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Memory Usage - N/A')
        
        # Training Progress (if available)
        if self.training_buffer:
            training_data = list(self.training_buffer)
            df_training = pd.DataFrame(training_data)
            
            if 'total_loss' in df_training.columns:
                axes[1, 1].plot(df_training['step'], df_training['total_loss'])
                axes[1, 1].set_title('Training Loss Over Time')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Training Progress - N/A')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Progress - N/A')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to: {save_path}")
        
        plt.show()
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.system_buffer:
            recommendations.append("Start monitoring to get recommendations")
            return recommendations
        
        # Analyze recent performance
        recent_data = list(self.system_buffer)[-100:]  # Last 100 samples
        
        avg_cpu = np.mean([d['cpu_percent'] for d in recent_data])
        avg_memory = np.mean([d['memory_percent'] for d in recent_data])
        
        # CPU recommendations
        if avg_cpu > 80:
            recommendations.append("High CPU usage detected - consider reducing batch size or DataLoader workers")
        elif avg_cpu < 30:
            recommendations.append("Low CPU usage - consider increasing batch size or DataLoader workers")
        
        # Memory recommendations
        if avg_memory > 85:
            recommendations.append("High memory usage - consider gradient accumulation or smaller batch sizes")
        elif avg_memory < 40:
            recommendations.append("Low memory usage - consider increasing batch size for better GPU utilization")
        
        # GPU recommendations
        if torch.cuda.is_available():
            recent_gpu_data = [d for d in recent_data if 'gpu_memory_percent' in d]
            if recent_gpu_data:
                avg_gpu_memory = np.mean([d['gpu_memory_percent'] for d in recent_gpu_data])
                
                if avg_gpu_memory > 90:
                    recommendations.append("High GPU memory usage - consider gradient accumulation or mixed precision")
                elif avg_gpu_memory < 50:
                    recommendations.append("Low GPU memory usage - consider increasing batch size or model size")
        
        # Training speed recommendations
        if self.training_buffer:
            training_data = list(self.training_buffer)
            if len(training_data) > 10:
                recent_training = training_data[-10:]
                time_diffs = [
                    recent_training[i]['timestamp'] - recent_training[i-1]['timestamp']
                    for i in range(1, len(recent_training))
                ]
                avg_step_time = np.mean(time_diffs)
                
                if avg_step_time > 5.0:  # 5 seconds per step
                    recommendations.append("Slow training detected - consider model optimization or hardware upgrade")
        
        # Alert-based recommendations
        if len(self.alerts) > 10:
            recommendations.append("Many performance alerts - review system configuration and resource allocation")
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        logger.info("Performance Monitor cleaned up")


def create_performance_dashboard(monitor: PerformanceMonitor, refresh_interval: int = 5):
    """Create a simple performance dashboard"""
    try:
        from IPython.display import clear_output
        import time
        
        while True:
            clear_output(wait=True)
            
            # Get current status
            summary = monitor.get_performance_summary()
            
            print("="*60)
            print("🖥️  TRAINING INFRASTRUCTURE PERFORMANCE DASHBOARD")
            print("="*60)
            
            if 'current_status' in summary:
                status = summary['current_status']
                print(f"🔄 Uptime: {status['uptime_hours']:.1f} hours")
                print(f"🧠 CPU: {status['cpu_percent']:.1f}%")
                print(f"💾 Memory: {status['memory_percent']:.1f}%")
                print(f"🎮 GPU Memory: {status.get('gpu_memory_percent', 0):.1f}%")
            
            if 'training_progress' in summary:
                progress = summary['training_progress']
                print(f"📈 Current Episode: {progress['current_episode']}")
                print(f"📊 Total Steps: {progress['total_training_steps']}")
            
            if 'recent_alerts' in summary and summary['recent_alerts']:
                print(f"⚠️  Recent Alerts: {len(summary['recent_alerts'])}")
                for alert in summary['recent_alerts'][-3:]:
                    print(f"   - {alert}")
            
            # Recommendations
            recommendations = monitor.get_optimization_recommendations()
            if recommendations:
                print("\n💡 Optimization Recommendations:")
                for rec in recommendations[:3]:
                    print(f"   • {rec}")
            
            print(f"\n🔄 Refreshing in {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")


if __name__ == "__main__":
    # Demo usage
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring(interval=1.0)
    
    # Simulate training
    for episode in range(5):
        for step in range(10):
            # Simulate training metrics
            metrics = {
                'actor_loss': np.random.uniform(0.1, 2.0),
                'critic_loss': np.random.uniform(0.1, 2.0),
                'total_loss': np.random.uniform(0.2, 4.0),
                'reward': np.random.uniform(-10, 10)
            }
            
            monitor.log_training_step(episode, step, metrics)
            time.sleep(0.1)
    
    # Generate report
    summary = monitor.get_performance_summary()
    print("Performance Summary:", json.dumps(summary, indent=2, default=str))
    
    # Save report
    monitor.save_performance_report()
    
    # Plot metrics
    monitor.plot_performance_metrics()
    
    # Get recommendations
    recommendations = monitor.get_optimization_recommendations()
    print("Recommendations:", recommendations)
    
    # Cleanup
    monitor.cleanup()