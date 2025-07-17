"""
Advanced Training Progress Monitor for Long-Running Processes
Real-time monitoring, visualization, alerting, and performance tracking
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import json
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import websocket
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    LOSS = "loss"
    ACCURACY = "accuracy"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    TEMPERATURE = "temperature"
    CUSTOM = "custom"

@dataclass
class TrainingMetric:
    """Single training metric data point"""
    name: str
    value: float
    timestamp: datetime
    epoch: int
    step: int
    metric_type: MetricType
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """Training alert"""
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False

@dataclass
class MonitoringConfig:
    """Configuration for training monitoring"""
    # Basic monitoring
    update_frequency: float = 1.0  # seconds
    history_window: int = 1000  # number of data points
    save_frequency: int = 100  # save metrics every N steps
    
    # Visualization
    enable_live_plots: bool = True
    plot_update_interval: float = 5.0  # seconds
    max_plot_points: int = 500
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown: float = 300.0  # seconds
    email_alerts: bool = False
    webhook_alerts: bool = False
    
    # Performance monitoring
    track_system_metrics: bool = True
    track_gpu_metrics: bool = True
    memory_threshold_gb: float = 8.0
    
    # Persistence
    save_to_disk: bool = True
    metrics_file: str = "training_metrics.json"
    log_file: str = "training_monitor.log"
    
    # Web interface
    enable_web_interface: bool = False
    web_port: int = 8080
    
    # Notification settings
    email_config: Dict[str, str] = None
    webhook_url: str = None
    slack_webhook: str = None

class MetricTracker:
    """Track individual metric with statistics"""
    
    def __init__(self, name: str, metric_type: MetricType, window_size: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.window_size = window_size
        
        # Data storage
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.epochs = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        
        # Statistics
        self.current_value = 0.0
        self.best_value = float('inf') if metric_type == MetricType.LOSS else float('-inf')
        self.worst_value = float('-inf') if metric_type == MetricType.LOSS else float('inf')
        self.total_updates = 0
        self.last_update = None
        
        # Trend analysis
        self.trend_window = 50
        self.trend_direction = "stable"
        self.trend_strength = 0.0
        
    def update(self, value: float, timestamp: datetime, epoch: int, step: int):
        """Update metric with new value"""
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.epochs.append(epoch)
        self.steps.append(step)
        
        self.current_value = value
        self.last_update = timestamp
        self.total_updates += 1
        
        # Update best/worst
        if self.metric_type == MetricType.LOSS:
            if value < self.best_value:
                self.best_value = value
            if value > self.worst_value:
                self.worst_value = value
        else:
            if value > self.best_value:
                self.best_value = value
            if value < self.worst_value:
                self.worst_value = value
        
        # Update trend
        self._update_trend()
    
    def _update_trend(self):
        """Update trend analysis"""
        if len(self.values) < self.trend_window:
            return
        
        recent_values = list(self.values)[-self.trend_window:]
        x = np.arange(len(recent_values))
        
        # Linear regression for trend
        slope, _ = np.polyfit(x, recent_values, 1)
        
        # Determine trend direction
        if abs(slope) < 1e-6:
            self.trend_direction = "stable"
        elif slope > 0:
            self.trend_direction = "increasing"
        else:
            self.trend_direction = "decreasing"
        
        # Calculate trend strength (normalized)
        value_range = max(recent_values) - min(recent_values)
        if value_range > 0:
            self.trend_strength = abs(slope) / value_range
        else:
            self.trend_strength = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self.values:
            return {}
        
        values_array = np.array(self.values)
        
        return {
            'current': self.current_value,
            'best': self.best_value,
            'worst': self.worst_value,
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'median': np.median(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'total_updates': self.total_updates,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def get_recent_data(self, n: int = 100) -> Dict[str, List]:
        """Get recent data points"""
        n = min(n, len(self.values))
        return {
            'values': list(self.values)[-n:],
            'timestamps': [t.isoformat() for t in list(self.timestamps)[-n:]],
            'epochs': list(self.epochs)[-n:],
            'steps': list(self.steps)[-n:]
        }

class AlertManager:
    """Manage training alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts = []
        self.alert_rules = {}
        self.last_alert_time = {}
        self.notification_queue = queue.Queue()
        
        # Start notification thread
        if config.enable_alerts:
            self.notification_thread = threading.Thread(target=self._notification_worker)
            self.notification_thread.daemon = True
            self.notification_thread.start()
    
    def add_alert_rule(self, 
                      metric_name: str,
                      threshold: float,
                      condition: str = "greater",
                      level: AlertLevel = AlertLevel.WARNING,
                      message: str = None):
        """Add alert rule for metric"""
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'condition': condition,
            'level': level,
            'message': message or f"{metric_name} {condition} {threshold}",
            'enabled': True
        }
        
        logger.info(f"Added alert rule: {metric_name} {condition} {threshold}")
    
    def check_alerts(self, metric: TrainingMetric):
        """Check if metric triggers any alerts"""
        if metric.name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric.name]
        if not rule['enabled']:
            return
        
        # Check cooldown
        cooldown_key = f"{metric.name}_{rule['level'].value}"
        if cooldown_key in self.last_alert_time:
            time_since_last = (datetime.now() - self.last_alert_time[cooldown_key]).total_seconds()
            if time_since_last < self.config.alert_cooldown:
                return
        
        # Check condition
        triggered = False
        if rule['condition'] == "greater" and metric.value > rule['threshold']:
            triggered = True
        elif rule['condition'] == "less" and metric.value < rule['threshold']:
            triggered = True
        elif rule['condition'] == "equal" and abs(metric.value - rule['threshold']) < 1e-6:
            triggered = True
        
        if triggered:
            alert = Alert(
                level=rule['level'],
                message=rule['message'],
                timestamp=metric.timestamp,
                metric_name=metric.name,
                metric_value=metric.value,
                threshold=rule['threshold']
            )
            
            self.alerts.append(alert)
            self.last_alert_time[cooldown_key] = datetime.now()
            
            # Queue for notification
            self.notification_queue.put(alert)
            
            logger.warning(f"Alert triggered: {alert.message}")
    
    def _notification_worker(self):
        """Worker thread for sending notifications"""
        while True:
            try:
                alert = self.notification_queue.get(timeout=1.0)
                self._send_notifications(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Notification error: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        # Email notification
        if self.config.email_alerts and self.config.email_config:
            self._send_email_alert(alert)
        
        # Webhook notification
        if self.config.webhook_alerts and self.config.webhook_url:
            self._send_webhook_alert(alert)
        
        # Slack notification
        if self.config.slack_webhook:
            self._send_slack_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            email_config = self.config.email_config
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = email_config['to']
            msg['Subject'] = f"Training Alert: {alert.level.value.upper()}"
            
            body = f"""
            Training Alert
            
            Level: {alert.level.value.upper()}
            Message: {alert.message}
            Metric: {alert.metric_name} = {alert.metric_value}
            Threshold: {alert.threshold}
            Time: {alert.timestamp.isoformat()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            payload = {
                'level': alert.level.value,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert.message}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            color = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }[alert.level]
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"Training Alert: {alert.level.value.upper()}",
                    'text': alert.message,
                    'fields': [
                        {
                            'title': 'Metric',
                            'value': f"{alert.metric_name} = {alert.metric_value:.4f}",
                            'short': True
                        },
                        {
                            'title': 'Threshold',
                            'value': f"{alert.threshold:.4f}",
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        }
                    ]
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.message}")
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class SystemMetricsCollector:
    """Collect system and GPU metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.gpu_available = torch.cuda.is_available()
        self.collect_system = config.track_system_metrics
        self.collect_gpu = config.track_gpu_metrics and self.gpu_available
        
    def collect_metrics(self) -> Dict[str, float]:
        """Collect all system metrics"""
        metrics = {}
        
        if self.collect_system:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = (disk.used / disk.total) * 100
            metrics['disk_used_gb'] = disk.used / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['network_sent_mb'] = network.bytes_sent / (1024**2)
            metrics['network_recv_mb'] = network.bytes_recv / (1024**2)
        
        if self.collect_gpu:
            # GPU metrics
            for i in range(torch.cuda.device_count()):
                try:
                    metrics[f'gpu_{i}_memory_allocated_mb'] = torch.cuda.memory_allocated(i) / (1024**2)
                    metrics[f'gpu_{i}_memory_cached_mb'] = torch.cuda.memory_reserved(i) / (1024**2)
                    metrics[f'gpu_{i}_memory_percent'] = (torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i)) * 100 if torch.cuda.max_memory_allocated(i) > 0 else 0
                    
                    # GPU utilization (requires nvidia-ml-py)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f'gpu_{i}_utilization'] = util.gpu
                        
                        # GPU temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f'gpu_{i}_temperature'] = temp
                        
                    except ImportError:
                        pass  # pynvml not available
                    except Exception as e:
                        logger.debug(f"GPU metrics error: {e}")
                        
                except Exception as e:
                    logger.debug(f"GPU {i} metrics error: {e}")
        
        return metrics

class TrainingProgressMonitor:
    """
    Comprehensive training progress monitor
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = {}
        self.system_metrics = {}
        self.start_time = datetime.now()
        
        # Initialize components
        self.alert_manager = AlertManager(config)
        self.system_collector = SystemMetricsCollector(config)
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.last_save = datetime.now()
        
        # Performance tracking
        self.training_phases = []
        self.current_phase = None
        
        # Live plotting
        self.live_plots = {}
        self.plot_figures = {}
        
        # Persistence
        self.metrics_file = Path(config.metrics_file)
        self.log_file = Path(config.log_file)
        
        # Load existing metrics
        self._load_metrics()
        
        logger.info("Training progress monitor initialized")
    
    def _load_metrics(self):
        """Load existing metrics from disk"""
        if self.config.save_to_disk and self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    # TODO: Reconstruct MetricTracker objects from saved data
                logger.info(f"Loaded existing metrics from {self.metrics_file}")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        if not self.config.save_to_disk:
            return
        
        try:
            # Prepare data for saving
            data = {
                'start_time': self.start_time.isoformat(),
                'metrics': {},
                'system_metrics': {},
                'training_phases': self.training_phases,
                'alerts': [asdict(alert) for alert in self.alert_manager.alerts]
            }
            
            # Save metric statistics
            for name, tracker in self.metrics.items():
                data['metrics'][name] = {
                    'statistics': tracker.get_statistics(),
                    'recent_data': tracker.get_recent_data(100)
                }
            
            # Save system metrics
            for name, tracker in self.system_metrics.items():
                data['system_metrics'][name] = {
                    'statistics': tracker.get_statistics(),
                    'recent_data': tracker.get_recent_data(100)
                }
            
            # Write to file
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.last_save = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Training monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Final save
        self._save_metrics()
        
        logger.info("Training monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self.system_collector.collect_metrics()
                current_time = datetime.now()
                
                # Update system metrics
                for name, value in system_metrics.items():
                    if name not in self.system_metrics:
                        self.system_metrics[name] = MetricTracker(
                            name, MetricType.CUSTOM, self.config.history_window
                        )
                    
                    self.system_metrics[name].update(value, current_time, 0, 0)
                
                # Check for memory alerts
                if 'memory_used_gb' in system_metrics:
                    memory_usage = system_metrics['memory_used_gb']
                    if memory_usage > self.config.memory_threshold_gb:
                        self.alert_manager.check_alerts(
                            TrainingMetric(
                                name='memory_usage',
                                value=memory_usage,
                                timestamp=current_time,
                                epoch=0,
                                step=0,
                                metric_type=MetricType.MEMORY_USAGE
                            )
                        )
                
                # Periodic save
                if (current_time - self.last_save).total_seconds() > self.config.save_frequency:
                    self._save_metrics()
                
                # Sleep until next update
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def log_metric(self, 
                  name: str,
                  value: float,
                  epoch: int,
                  step: int,
                  metric_type: MetricType = MetricType.CUSTOM,
                  metadata: Dict[str, Any] = None):
        """Log a training metric"""
        current_time = datetime.now()
        
        # Create or update metric tracker
        if name not in self.metrics:
            self.metrics[name] = MetricTracker(name, metric_type, self.config.history_window)
        
        self.metrics[name].update(value, current_time, epoch, step)
        
        # Create metric object
        metric = TrainingMetric(
            name=name,
            value=value,
            timestamp=current_time,
            epoch=epoch,
            step=step,
            metric_type=metric_type,
            metadata=metadata
        )
        
        # Check alerts
        self.alert_manager.check_alerts(metric)
        
        # Log to file
        log_entry = f"{current_time.isoformat()} | {name}: {value:.6f} | Epoch: {epoch} | Step: {step}"
        if metadata:
            log_entry += f" | {json.dumps(metadata)}"
        
        logger.info(log_entry)
    
    def log_multiple_metrics(self, 
                           metrics: Dict[str, float],
                           epoch: int,
                           step: int,
                           metadata: Dict[str, Any] = None):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            # Infer metric type from name
            metric_type = MetricType.CUSTOM
            if 'loss' in name.lower():
                metric_type = MetricType.LOSS
            elif 'accuracy' in name.lower() or 'acc' in name.lower():
                metric_type = MetricType.ACCURACY
            elif 'lr' in name.lower() or 'learning_rate' in name.lower():
                metric_type = MetricType.LEARNING_RATE
            elif 'grad' in name.lower():
                metric_type = MetricType.GRADIENT_NORM
            elif 'memory' in name.lower():
                metric_type = MetricType.MEMORY_USAGE
            
            self.log_metric(name, value, epoch, step, metric_type, metadata)
    
    def start_phase(self, phase_name: str, description: str = None):
        """Start a new training phase"""
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = {
            'name': phase_name,
            'description': description,
            'start_time': datetime.now(),
            'start_metrics': {name: tracker.current_value for name, tracker in self.metrics.items()}
        }
        
        logger.info(f"Started training phase: {phase_name}")
    
    def end_phase(self):
        """End current training phase"""
        if not self.current_phase:
            return
        
        phase = self.current_phase.copy()
        phase['end_time'] = datetime.now()
        phase['duration'] = (phase['end_time'] - phase['start_time']).total_seconds()
        phase['end_metrics'] = {name: tracker.current_value for name, tracker in self.metrics.items()}
        
        self.training_phases.append(phase)
        self.current_phase = None
        
        logger.info(f"Ended training phase: {phase['name']} (duration: {phase['duration']:.2f}s)")
    
    def add_alert_rule(self, metric_name: str, threshold: float, condition: str = "greater"):
        """Add alert rule for metric"""
        self.alert_manager.add_alert_rule(metric_name, threshold, condition)
    
    def create_live_plot(self, metric_names: List[str], title: str = "Training Progress"):
        """Create live plot for metrics"""
        if not self.config.enable_live_plots:
            return
        
        try:
            plt.ion()  # Turn on interactive mode
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(title)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Store figure and axis
            plot_id = f"live_plot_{len(self.live_plots)}"
            self.live_plots[plot_id] = {
                'fig': fig,
                'ax': ax,
                'metric_names': metric_names,
                'lines': {},
                'title': title
            }
            
            # Create lines for each metric
            for metric_name in metric_names:
                line, = ax.plot([], [], label=metric_name, linewidth=2)
                self.live_plots[plot_id]['lines'][metric_name] = line
            
            ax.legend()
            plt.tight_layout()
            plt.show()
            
            return plot_id
            
        except Exception as e:
            logger.error(f"Failed to create live plot: {e}")
            return None
    
    def update_live_plots(self):
        """Update all live plots"""
        if not self.config.enable_live_plots:
            return
        
        for plot_id, plot_info in self.live_plots.items():
            try:
                ax = plot_info['ax']
                
                # Update each metric line
                for metric_name in plot_info['metric_names']:
                    if metric_name in self.metrics:
                        tracker = self.metrics[metric_name]
                        recent_data = tracker.get_recent_data(self.config.max_plot_points)
                        
                        if recent_data['values']:
                            line = plot_info['lines'][metric_name]
                            line.set_data(recent_data['steps'], recent_data['values'])
                
                # Auto-scale axes
                ax.relim()
                ax.autoscale_view()
                
                # Update plot
                plot_info['fig'].canvas.draw()
                plot_info['fig'].canvas.flush_events()
                
            except Exception as e:
                logger.error(f"Error updating live plot {plot_id}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        current_time = datetime.now()
        total_duration = (current_time - self.start_time).total_seconds()
        
        summary = {
            'start_time': self.start_time.isoformat(),
            'current_time': current_time.isoformat(),
            'total_duration_seconds': total_duration,
            'total_duration_human': str(timedelta(seconds=int(total_duration))),
            'metrics_summary': {},
            'system_metrics_summary': {},
            'training_phases': self.training_phases,
            'alerts': {
                'total_alerts': len(self.alert_manager.alerts),
                'active_alerts': len([a for a in self.alert_manager.alerts if not a.resolved]),
                'recent_alerts': [asdict(a) for a in self.alert_manager.alerts[-5:]]
            }
        }
        
        # Add metric summaries
        for name, tracker in self.metrics.items():
            summary['metrics_summary'][name] = tracker.get_statistics()
        
        # Add system metric summaries
        for name, tracker in self.system_metrics.items():
            summary['system_metrics_summary'][name] = tracker.get_statistics()
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file"""
        summary = self.get_training_summary()
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format.lower() == "csv":
            # Create DataFrame for CSV export
            rows = []
            for name, tracker in self.metrics.items():
                recent_data = tracker.get_recent_data(1000)
                for i, value in enumerate(recent_data['values']):
                    rows.append({
                        'metric_name': name,
                        'value': value,
                        'timestamp': recent_data['timestamps'][i],
                        'epoch': recent_data['epochs'][i],
                        'step': recent_data['steps'][i]
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        
        # Close plots
        for plot_info in self.live_plots.values():
            try:
                plt.close(plot_info['fig'])
            except:
                pass
        
        logger.info("Training monitor cleanup completed")


def create_monitoring_config(
    enable_live_plots: bool = True,
    enable_alerts: bool = True,
    save_to_disk: bool = True,
    track_system_metrics: bool = True
) -> MonitoringConfig:
    """Create optimized monitoring configuration"""
    
    config = MonitoringConfig(
        enable_live_plots=enable_live_plots,
        enable_alerts=enable_alerts,
        save_to_disk=save_to_disk,
        track_system_metrics=track_system_metrics,
        track_gpu_metrics=torch.cuda.is_available(),
        update_frequency=1.0,
        plot_update_interval=5.0,
        history_window=2000,
        memory_threshold_gb=8.0
    )
    
    return config


# Example usage functions
def setup_basic_monitoring(model_name: str = "training") -> TrainingProgressMonitor:
    """Set up basic training monitoring"""
    config = create_monitoring_config()
    monitor = TrainingProgressMonitor(config)
    monitor.start_monitoring()
    
    # Add common alert rules
    monitor.add_alert_rule("loss", 10.0, "greater")
    monitor.add_alert_rule("memory_usage", 8.0, "greater")
    
    return monitor


def setup_advanced_monitoring(
    model_name: str = "training",
    email_config: Dict[str, str] = None,
    slack_webhook: str = None
) -> TrainingProgressMonitor:
    """Set up advanced training monitoring with notifications"""
    
    config = create_monitoring_config()
    config.email_alerts = email_config is not None
    config.email_config = email_config
    config.slack_webhook = slack_webhook
    
    monitor = TrainingProgressMonitor(config)
    monitor.start_monitoring()
    
    # Add comprehensive alert rules
    monitor.add_alert_rule("loss", 10.0, "greater")
    monitor.add_alert_rule("gradient_norm", 5.0, "greater")
    monitor.add_alert_rule("memory_usage", 8.0, "greater")
    monitor.add_alert_rule("gpu_0_temperature", 80.0, "greater")
    
    # Create live plots
    monitor.create_live_plot(["loss", "validation_loss"], "Training Loss")
    monitor.create_live_plot(["accuracy", "validation_accuracy"], "Training Accuracy")
    
    return monitor