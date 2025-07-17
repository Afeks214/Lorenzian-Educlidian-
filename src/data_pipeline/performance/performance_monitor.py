"""
Performance monitoring for data pipeline
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np
from pathlib import Path
import json
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_bytes: int
    disk_io_read_bytes: int
    disk_io_write_bytes: int
    network_io_sent_bytes: int
    network_io_recv_bytes: int
    active_threads: int
    open_files: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_bytes': self.memory_bytes,
            'disk_io_read_bytes': self.disk_io_read_bytes,
            'disk_io_write_bytes': self.disk_io_write_bytes,
            'network_io_sent_bytes': self.network_io_sent_bytes,
            'network_io_recv_bytes': self.network_io_recv_bytes,
            'active_threads': self.active_threads,
            'open_files': self.open_files
        }

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    monitoring_interval: float = 1.0  # seconds
    history_size: int = 1000
    enable_disk_io_monitoring: bool = True
    enable_network_io_monitoring: bool = True
    enable_thread_monitoring: bool = True
    enable_file_monitoring: bool = True
    enable_alerts: bool = True
    
    # Alert thresholds
    cpu_threshold: float = 80.0  # percent
    memory_threshold: float = 85.0  # percent
    disk_io_threshold: int = 100 * 1024 * 1024  # 100MB/s
    
    # Persistence
    enable_persistence: bool = True
    db_path: str = "/tmp/performance_monitor.db"

class PerformanceMonitor:
    """
    Comprehensive performance monitor for data pipeline
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Performance history
        self.metrics_history: deque = deque(maxlen=self.config.history_size)
        self.alerts: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Process info
        self.process = psutil.Process()
        self.initial_disk_io = self.process.io_counters() if self.config.enable_disk_io_monitoring else None
        self.initial_network_io = psutil.net_io_counters() if self.config.enable_network_io_monitoring else None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup persistence
        if self.config.enable_persistence:
            self._setup_persistence()
    
    def _setup_persistence(self):
        """Setup performance history persistence"""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_bytes INTEGER,
                    disk_io_read_bytes INTEGER,
                    disk_io_write_bytes INTEGER,
                    network_io_sent_bytes INTEGER,
                    network_io_recv_bytes INTEGER,
                    active_threads INTEGER,
                    open_files INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_metrics(timestamp)
            """)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(metrics)
                
                # Persist metrics
                if self.config.enable_persistence:
                    self._persist_metrics(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            disk_io_read_bytes = 0
            disk_io_write_bytes = 0
            if self.config.enable_disk_io_monitoring:
                try:
                    io_counters = self.process.io_counters()
                    if self.initial_disk_io:
                        disk_io_read_bytes = io_counters.read_bytes - self.initial_disk_io.read_bytes
                        disk_io_write_bytes = io_counters.write_bytes - self.initial_disk_io.write_bytes
                    else:
                        disk_io_read_bytes = io_counters.read_bytes
                        disk_io_write_bytes = io_counters.write_bytes
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            # Network I/O
            network_io_sent_bytes = 0
            network_io_recv_bytes = 0
            if self.config.enable_network_io_monitoring:
                try:
                    net_io = psutil.net_io_counters()
                    if self.initial_network_io:
                        network_io_sent_bytes = net_io.bytes_sent - self.initial_network_io.bytes_sent
                        network_io_recv_bytes = net_io.bytes_recv - self.initial_network_io.bytes_recv
                    else:
                        network_io_sent_bytes = net_io.bytes_sent
                        network_io_recv_bytes = net_io.bytes_recv
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            # Thread and file info
            active_threads = 0
            open_files = 0
            if self.config.enable_thread_monitoring:
                try:
                    active_threads = self.process.num_threads()
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            if self.config.enable_file_monitoring:
                try:
                    open_files = self.process.num_fds()
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_bytes=memory_info.rss,
                disk_io_read_bytes=disk_io_read_bytes,
                disk_io_write_bytes=disk_io_write_bytes,
                network_io_sent_bytes=network_io_sent_bytes,
                network_io_recv_bytes=network_io_recv_bytes,
                active_threads=active_threads,
                open_files=open_files
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_bytes=0,
                disk_io_read_bytes=0,
                disk_io_write_bytes=0,
                network_io_sent_bytes=0,
                network_io_recv_bytes=0,
                active_threads=0,
                open_files=0
            )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.config.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'message': f'CPU usage high: {metrics.cpu_percent:.1f}%',
                'threshold': self.config.cpu_threshold,
                'current': metrics.cpu_percent,
                'timestamp': metrics.timestamp
            })
        
        # Memory alert
        if metrics.memory_percent > self.config.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'message': f'Memory usage high: {metrics.memory_percent:.1f}%',
                'threshold': self.config.memory_threshold,
                'current': metrics.memory_percent,
                'timestamp': metrics.timestamp
            })
        
        # Disk I/O alert
        if metrics.disk_io_write_bytes > self.config.disk_io_threshold:
            alerts.append({
                'type': 'disk_io_high',
                'message': f'Disk I/O high: {metrics.disk_io_write_bytes / 1024 / 1024:.1f} MB/s',
                'threshold': self.config.disk_io_threshold,
                'current': metrics.disk_io_write_bytes,
                'timestamp': metrics.timestamp
            })
        
        # Store alerts
        if alerts:
            with self._lock:
                self.alerts.extend(alerts)
                
                # Keep only recent alerts (last 100)
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
            
            # Log alerts
            for alert in alerts:
                logger.warning(alert['message'])
    
    def _persist_metrics(self, metrics: PerformanceMetrics):
        """Persist metrics to database"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_bytes, 
                     disk_io_read_bytes, disk_io_write_bytes, 
                     network_io_sent_bytes, network_io_recv_bytes, 
                     active_threads, open_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_bytes,
                    metrics.disk_io_read_bytes,
                    metrics.disk_io_write_bytes,
                    metrics.network_io_sent_bytes,
                    metrics.network_io_recv_bytes,
                    metrics.active_threads,
                    metrics.open_files
                ))
        except Exception as e:
            logger.error(f"Error persisting metrics: {str(e)}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[PerformanceMetrics]:
        """Get metrics history"""
        with self._lock:
            if duration_seconds is None:
                return list(self.metrics_history)
            
            cutoff_time = time.time() - duration_seconds
            return [
                metrics for metrics in self.metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def get_performance_summary(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get performance summary statistics"""
        metrics_list = self.get_metrics_history(duration_seconds)
        
        if not metrics_list:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics_list]
        memory_values = [m.memory_percent for m in metrics_list]
        memory_bytes = [m.memory_bytes for m in metrics_list]
        
        return {
            'duration_seconds': duration_seconds,
            'sample_count': len(metrics_list),
            'cpu_stats': {
                'avg': np.mean(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_stats': {
                'avg_percent': np.mean(memory_values),
                'min_percent': np.min(memory_values),
                'max_percent': np.max(memory_values),
                'avg_bytes': np.mean(memory_bytes),
                'peak_bytes': np.max(memory_bytes)
            },
            'disk_io_stats': {
                'total_read_bytes': sum(m.disk_io_read_bytes for m in metrics_list),
                'total_write_bytes': sum(m.disk_io_write_bytes for m in metrics_list),
                'avg_read_rate': np.mean([m.disk_io_read_bytes for m in metrics_list]),
                'avg_write_rate': np.mean([m.disk_io_write_bytes for m in metrics_list])
            },
            'network_io_stats': {
                'total_sent_bytes': sum(m.network_io_sent_bytes for m in metrics_list),
                'total_recv_bytes': sum(m.network_io_recv_bytes for m in metrics_list)
            },
            'thread_stats': {
                'avg_threads': np.mean([m.active_threads for m in metrics_list]),
                'max_threads': np.max([m.active_threads for m in metrics_list])
            }
        }
    
    def get_alerts(self, duration_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        with self._lock:
            if duration_seconds is None:
                return self.alerts.copy()
            
            cutoff_time = time.time() - duration_seconds
            return [
                alert for alert in self.alerts
                if alert['timestamp'] >= cutoff_time
            ]
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self._lock:
            self.alerts.clear()
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """Export metrics to file"""
        metrics_data = [m.to_dict() for m in self.get_metrics_history()]
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {file_path}")
    
    def reset_metrics(self):
        """Reset metrics history"""
        with self._lock:
            self.metrics_history.clear()
            self.alerts.clear()
        
        logger.info("Metrics history reset")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': {
                path: psutil.disk_usage(path)._asdict() 
                for path in ['/tmp', '/']
                if Path(path).exists()
            },
            'network_interfaces': list(psutil.net_if_addrs().keys()),
            'boot_time': psutil.boot_time(),
            'process_count': len(psutil.pids())
        }


@contextmanager
def performance_monitoring(config: Optional[PerformanceConfig] = None):
    """Context manager for performance monitoring"""
    monitor = PerformanceMonitor(config)
    
    try:
        monitor.start_monitoring()
        yield monitor
    finally:
        monitor.stop_monitoring()


class PerformanceProfiler:
    """Advanced performance profiler"""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code block"""
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            with self._lock:
                if name not in self.profiles:
                    self.profiles[name] = []
                
                self.profiles[name].append(elapsed)
    
    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """Get profile statistics"""
        with self._lock:
            if name not in self.profiles:
                return {}
            
            times = self.profiles[name]
            return {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get all profile statistics"""
        return {
            name: self.get_profile_stats(name)
            for name in self.profiles.keys()
        }
    
    def clear_profiles(self):
        """Clear all profiles"""
        with self._lock:
            self.profiles.clear()


# Global profiler instance
profiler = PerformanceProfiler()

def profile_function(name: str):
    """Decorator for profiling functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with profiler.profile(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator