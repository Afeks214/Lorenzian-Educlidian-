#!/usr/bin/env python3
"""
Comprehensive Training Performance Monitoring System
Tracks metrics, resource usage, and system health during training
"""

import os
import json
import time
import psutil
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import GPUtil
import torch
import numpy as np
from collections import defaultdict, deque

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_temperature: List[float]
    network_io_sent: int
    network_io_recv: int

@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    batch_size: int
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time_per_step: Optional[float] = None

class TrainingMonitor:
    """Comprehensive training monitoring system"""
    
    def __init__(self, log_dir: str = "/home/QuantNova/GrandModel/colab/logs/performance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=10000)
        self.training_metrics: deque = deque(maxlen=10000)
        self.alerts: List[Dict] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_utilization': 95.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 85.0,
            'training_time_per_step': 10.0,  # seconds
            'gradient_norm': 100.0
        }
        
        # Metrics aggregation
        self.metrics_history = defaultdict(list)
        
        self.logger.info("Training monitor initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.log_dir / f"training_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # GPU metrics
            gpu_utilization = []
            gpu_memory_used = []
            gpu_memory_total = []
            gpu_temperature = []
            
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_utilization.append(gpu.load * 100)
                        gpu_memory_used.append(gpu.memoryUsed)
                        gpu_memory_total.append(gpu.memoryTotal)
                        gpu_temperature.append(gpu.temperature)
                except Exception as e:
                    self.logger.warning(f"GPU metrics collection failed: {e}")
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_temperature=gpu_temperature,
                network_io_sent=net_io.bytes_sent,
                network_io_recv=net_io.bytes_recv
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        try:
            self.training_metrics.append(metrics)
            self.metrics_history['loss'].append(metrics.loss)
            self.metrics_history['learning_rate'].append(metrics.learning_rate)
            
            if metrics.validation_loss is not None:
                self.metrics_history['validation_loss'].append(metrics.validation_loss)
            
            self.logger.info(f"Epoch {metrics.epoch}, Step {metrics.step}: "
                           f"Loss={metrics.loss:.4f}, LR={metrics.learning_rate:.6f}")
            
            # Check for anomalies
            self.check_training_anomalies(metrics)
            
        except Exception as e:
            self.logger.error(f"Error logging training metrics: {e}")
    
    def check_training_anomalies(self, metrics: TrainingMetrics):
        """Check for training anomalies and alert"""
        alerts = []
        
        # Check gradient norm
        if metrics.gradient_norm and metrics.gradient_norm > self.thresholds['gradient_norm']:
            alerts.append({
                'type': 'gradient_explosion',
                'severity': 'high',
                'message': f"Gradient norm {metrics.gradient_norm:.2f} exceeds threshold {self.thresholds['gradient_norm']}",
                'timestamp': time.time()
            })
        
        # Check training time per step
        if metrics.training_time_per_step and metrics.training_time_per_step > self.thresholds['training_time_per_step']:
            alerts.append({
                'type': 'slow_training',
                'severity': 'medium',
                'message': f"Training time per step {metrics.training_time_per_step:.2f}s exceeds threshold {self.thresholds['training_time_per_step']}s",
                'timestamp': time.time()
            })
        
        # Check for NaN losses
        if np.isnan(metrics.loss) or np.isinf(metrics.loss):
            alerts.append({
                'type': 'nan_loss',
                'severity': 'critical',
                'message': f"Loss is NaN or infinite: {metrics.loss}",
                'timestamp': time.time()
            })
        
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {alert['message']}")
    
    def check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alerts"""
        if not metrics:
            return
        
        alerts = []
        
        # CPU usage
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'medium',
                'message': f"CPU usage {metrics.cpu_percent:.1f}% exceeds threshold {self.thresholds['cpu_percent']}%",
                'timestamp': time.time()
            })
        
        # Memory usage
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'high',
                'message': f"Memory usage {metrics.memory_percent:.1f}% exceeds threshold {self.thresholds['memory_percent']}%",
                'timestamp': time.time()
            })
        
        # GPU alerts
        for i, (util, mem_used, mem_total, temp) in enumerate(zip(
            metrics.gpu_utilization, metrics.gpu_memory_used, 
            metrics.gpu_memory_total, metrics.gpu_temperature
        )):
            mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            
            if util > self.thresholds['gpu_utilization']:
                alerts.append({
                    'type': 'high_gpu_utilization',
                    'severity': 'medium',
                    'message': f"GPU {i} utilization {util:.1f}% exceeds threshold {self.thresholds['gpu_utilization']}%",
                    'timestamp': time.time()
                })
            
            if mem_percent > self.thresholds['gpu_memory_percent']:
                alerts.append({
                    'type': 'high_gpu_memory',
                    'severity': 'high',
                    'message': f"GPU {i} memory {mem_percent:.1f}% exceeds threshold {self.thresholds['gpu_memory_percent']}%",
                    'timestamp': time.time()
                })
            
            if temp > self.thresholds['gpu_temperature']:
                alerts.append({
                    'type': 'high_gpu_temperature',
                    'severity': 'high',
                    'message': f"GPU {i} temperature {temp:.1f}°C exceeds threshold {self.thresholds['gpu_temperature']}°C",
                    'timestamp': time.time()
                })
        
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(f"SYSTEM ALERT: {alert['message']}")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.get_system_metrics()
                if metrics:
                    self.system_metrics.append(metrics)
                    self.check_system_alerts(metrics)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Stopped system monitoring")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'system_metrics': {
                'avg_cpu_percent': np.mean([m.cpu_percent for m in self.system_metrics]) if self.system_metrics else 0,
                'avg_memory_percent': np.mean([m.memory_percent for m in self.system_metrics]) if self.system_metrics else 0,
                'avg_gpu_utilization': np.mean([np.mean(m.gpu_utilization) for m in self.system_metrics if m.gpu_utilization]) if self.system_metrics else 0,
                'max_gpu_temperature': max([max(m.gpu_temperature) for m in self.system_metrics if m.gpu_temperature], default=0)
            },
            'training_metrics': {
                'total_steps': len(self.training_metrics),
                'avg_loss': np.mean(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
                'min_loss': min(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
                'current_learning_rate': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else 0
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a['severity'] == 'critical']),
                'high_alerts': len([a for a in self.alerts if a['severity'] == 'high']),
                'medium_alerts': len([a for a in self.alerts if a['severity'] == 'medium'])
            }
        }
        
        return summary
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save all metrics to file"""
        if not filename:
            filename = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        data = {
            'system_metrics': [asdict(m) for m in self.system_metrics],
            'training_metrics': [asdict(m) for m in self.training_metrics],
            'alerts': self.alerts,
            'performance_summary': self.get_performance_summary(),
            'thresholds': self.thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")
    
    def generate_report(self) -> str:
        """Generate performance report"""
        summary = self.get_performance_summary()
        
        report = f"""
Training Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM PERFORMANCE:
- Average CPU Usage: {summary['system_metrics']['avg_cpu_percent']:.1f}%
- Average Memory Usage: {summary['system_metrics']['avg_memory_percent']:.1f}%
- Average GPU Utilization: {summary['system_metrics']['avg_gpu_utilization']:.1f}%
- Maximum GPU Temperature: {summary['system_metrics']['max_gpu_temperature']:.1f}°C

TRAINING METRICS:
- Total Training Steps: {summary['training_metrics']['total_steps']}
- Average Loss: {summary['training_metrics']['avg_loss']:.4f}
- Minimum Loss: {summary['training_metrics']['min_loss']:.4f}
- Current Learning Rate: {summary['training_metrics']['current_learning_rate']:.6f}

ALERTS SUMMARY:
- Total Alerts: {summary['alerts']['total_alerts']}
- Critical Alerts: {summary['alerts']['critical_alerts']}
- High Severity Alerts: {summary['alerts']['high_alerts']}
- Medium Severity Alerts: {summary['alerts']['medium_alerts']}

RECENT ALERTS:
"""
        
        # Add recent alerts
        recent_alerts = sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
        for alert in recent_alerts:
            timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            report += f"- [{timestamp}] {alert['severity'].upper()}: {alert['message']}\n"
        
        return report

# Example usage and demo
if __name__ == "__main__":
    # Initialize monitor
    monitor = TrainingMonitor()
    
    # Start system monitoring
    monitor.start_monitoring(interval=10.0)
    
    try:
        # Simulate training loop
        for epoch in range(3):
            for step in range(10):
                # Simulate training metrics
                training_metrics = TrainingMetrics(
                    timestamp=time.time(),
                    epoch=epoch,
                    step=step,
                    loss=1.0 / (step + 1),
                    learning_rate=0.001 * (0.9 ** epoch),
                    batch_size=32,
                    gradient_norm=np.random.uniform(0.1, 2.0),
                    training_time_per_step=np.random.uniform(0.5, 2.0)
                )
                
                monitor.log_training_metrics(training_metrics)
                time.sleep(1)
    
    finally:
        # Stop monitoring and save results
        monitor.stop_monitoring()
        monitor.save_metrics()
        
        # Generate and print report
        report = monitor.generate_report()
        print(report)