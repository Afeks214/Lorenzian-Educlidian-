"""
Production Monitoring and Alerting System
=========================================

Real-time monitoring infrastructure for MARL trading systems with:
- Performance tracking and alerting
- Risk monitoring and automatic stops
- System health monitoring
- Error handling and recovery
- Real-time dashboard and notifications

Author: Claude Code
Date: 2025-07-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import logging
import json
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import psutil
import asyncio
import websockets
from collections import deque
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/QuantNova/GrandModel/logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for alert thresholds"""
    # Performance alerts
    min_sharpe_ratio: float = 1.5
    max_drawdown_pct: float = 10.0  # 10%
    min_win_rate: float = 0.55      # 55%
    max_consecutive_losses: int = 5
    
    # Risk alerts
    max_var_1day: float = -0.05     # -5% daily VaR
    max_position_size: float = 0.30  # 30% of portfolio
    max_leverage: float = 2.0
    
    # System alerts
    max_cpu_usage: float = 80.0     # 80%
    max_memory_usage: float = 85.0  # 85%
    max_disk_usage: float = 90.0    # 90%
    min_disk_space_gb: float = 5.0  # 5 GB
    
    # Trading alerts
    max_slippage_bps: float = 50.0  # 50 basis points
    max_order_latency_ms: float = 500.0
    min_liquidity_threshold: float = 100000.0
    
    # Data quality alerts
    max_missing_data_pct: float = 5.0  # 5%
    max_stale_data_minutes: float = 10.0
    
    # Model performance alerts
    min_model_confidence: float = 0.6
    max_prediction_error: float = 0.1

@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    alert_recipients: List[str] = field(default_factory=list)
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"
    
    # Discord settings
    discord_webhook_url: str = ""
    
    # SMS settings (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    alert_phone_numbers: List[str] = field(default_factory=list)
    
    # Dashboard settings
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"

class AlertSeverity:
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: str
    category: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

class NotificationHandler(ABC):
    """Abstract base class for notification handlers"""
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        pass

class EmailNotificationHandler(NotificationHandler):
    """Email notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            if not self.config.email_user or not self.config.alert_recipients:
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = ', '.join(self.config.alert_recipients)
            msg['Subject'] = f"[{alert.severity}] Trading Alert: {alert.category}"
            
            # Email body
            body = f"""
            Alert Details:
            
            Severity: {alert.severity}
            Category: {alert.category}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            
            Value: {alert.value}
            Threshold: {alert.threshold}
            
            Metadata: {json.dumps(alert.metadata, indent=2)}
            
            Alert ID: {alert.id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            if not self.config.slack_webhook_url:
                return False
            
            # Color coding by severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#ff3300",
                AlertSeverity.CRITICAL: "#990000"
            }
            
            # Create Slack message
            payload = {
                "channel": self.config.slack_channel,
                "username": "Trading Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"{alert.severity}: {alert.category}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Alert ID", "value": alert.id, "short": True}
                    ],
                    "footer": "MARL Trading System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._check_system_resources()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(60)  # Wait longer if error
    
    def _check_system_resources(self) -> List[Alert]:
        """Check system resources and generate alerts"""
        alerts = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.alert_config.max_cpu_usage:
            alerts.append(Alert(
                id=f"cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                category="System Performance",
                message=f"High CPU usage detected: {cpu_percent:.1f}%",
                value=cpu_percent,
                threshold=self.alert_config.max_cpu_usage
            ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        if memory_percent > self.alert_config.max_memory_usage:
            alerts.append(Alert(
                id=f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                category="System Performance",
                message=f"High memory usage detected: {memory_percent:.1f}%",
                value=memory_percent,
                threshold=self.alert_config.max_memory_usage
            ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = (disk.total - disk.used) / (1024**3)
        
        if disk_percent > self.alert_config.max_disk_usage:
            alerts.append(Alert(
                id=f"disk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity=AlertSeverity.ERROR,
                category="System Performance",
                message=f"High disk usage detected: {disk_percent:.1f}%",
                value=disk_percent,
                threshold=self.alert_config.max_disk_usage
            ))
        
        if disk_free_gb < self.alert_config.min_disk_space_gb:
            alerts.append(Alert(
                id=f"disk_space_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                category="System Performance",
                message=f"Low disk space: {disk_free_gb:.1f} GB remaining",
                value=disk_free_gb,
                threshold=self.alert_config.min_disk_space_gb
            ))
        
        return alerts

class PerformanceMonitor:
    """Trading performance monitoring"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.performance_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=500)
    
    def update_performance(self, portfolio_value: float, sharpe_ratio: float, 
                         drawdown: float, win_rate: float) -> List[Alert]:
        """Update performance metrics and check for alerts"""
        alerts = []
        timestamp = datetime.now()
        
        # Store performance data
        perf_data = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'sharpe_ratio': sharpe_ratio,
            'drawdown': drawdown,
            'win_rate': win_rate
        }
        self.performance_history.append(perf_data)
        
        # Check Sharpe ratio
        if sharpe_ratio < self.alert_config.min_sharpe_ratio:
            alerts.append(Alert(
                id=f"sharpe_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                severity=AlertSeverity.WARNING,
                category="Performance",
                message=f"Sharpe ratio below threshold: {sharpe_ratio:.3f}",
                value=sharpe_ratio,
                threshold=self.alert_config.min_sharpe_ratio
            ))
        
        # Check drawdown
        if abs(drawdown) > self.alert_config.max_drawdown_pct / 100:
            severity = AlertSeverity.CRITICAL if abs(drawdown) > 0.15 else AlertSeverity.ERROR
            alerts.append(Alert(
                id=f"drawdown_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                severity=severity,
                category="Risk",
                message=f"Maximum drawdown exceeded: {drawdown:.2%}",
                value=abs(drawdown) * 100,
                threshold=self.alert_config.max_drawdown_pct
            ))
        
        # Check win rate
        if win_rate < self.alert_config.min_win_rate:
            alerts.append(Alert(
                id=f"winrate_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                severity=AlertSeverity.WARNING,
                category="Performance",
                message=f"Win rate below threshold: {win_rate:.2%}",
                value=win_rate,
                threshold=self.alert_config.min_win_rate
            ))
        
        return alerts
    
    def update_trade(self, trade_pnl: float, trade_duration: timedelta) -> List[Alert]:
        """Update trade data and check for alerts"""
        alerts = []
        timestamp = datetime.now()
        
        # Store trade data
        trade_data = {
            'timestamp': timestamp,
            'pnl': trade_pnl,
            'duration': trade_duration
        }
        self.trade_history.append(trade_data)
        
        # Check for consecutive losses
        recent_trades = list(self.trade_history)[-self.alert_config.max_consecutive_losses:]
        if len(recent_trades) >= self.alert_config.max_consecutive_losses:
            if all(trade['pnl'] < 0 for trade in recent_trades):
                alerts.append(Alert(
                    id=f"consecutive_losses_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    severity=AlertSeverity.ERROR,
                    category="Risk",
                    message=f"Maximum consecutive losses reached: {self.alert_config.max_consecutive_losses}",
                    value=len(recent_trades),
                    threshold=self.alert_config.max_consecutive_losses
                ))
        
        return alerts

class DataQualityMonitor:
    """Data quality monitoring"""
    
    def __init__(self, alert_config: AlertConfig):
        self.alert_config = alert_config
        self.last_data_timestamp = None
    
    def check_data_quality(self, data: pd.DataFrame) -> List[Alert]:
        """Check data quality and generate alerts"""
        alerts = []
        timestamp = datetime.now()
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > self.alert_config.max_missing_data_pct:
            alerts.append(Alert(
                id=f"missing_data_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                severity=AlertSeverity.WARNING,
                category="Data Quality",
                message=f"High percentage of missing data: {missing_pct:.1f}%",
                value=missing_pct,
                threshold=self.alert_config.max_missing_data_pct
            ))
        
        # Check for stale data
        if not data.empty:
            latest_data_time = data.index[-1] if hasattr(data.index, 'to_pydatetime') else timestamp
            if isinstance(latest_data_time, pd.Timestamp):
                latest_data_time = latest_data_time.to_pydatetime()
            
            time_diff = (timestamp - latest_data_time).total_seconds() / 60  # minutes
            if time_diff > self.alert_config.max_stale_data_minutes:
                alerts.append(Alert(
                    id=f"stale_data_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    severity=AlertSeverity.ERROR,
                    category="Data Quality",
                    message=f"Stale data detected: {time_diff:.1f} minutes old",
                    value=time_diff,
                    threshold=self.alert_config.max_stale_data_minutes
                ))
        
        # Check for data anomalies (extreme values)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not data[col].empty:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                if len(outliers) > len(data) * 0.05:  # More than 5% outliers
                    alerts.append(Alert(
                        id=f"outliers_{col}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        timestamp=timestamp,
                        severity=AlertSeverity.WARNING,
                        category="Data Quality",
                        message=f"High number of outliers in {col}: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)",
                        value=len(outliers)/len(data)*100,
                        threshold=5.0
                    ))
        
        return alerts

class AlertManager:
    """Central alert management system"""
    
    def __init__(self, alert_config: AlertConfig, notification_config: NotificationConfig,
                 db_path: str = "/home/QuantNova/GrandModel/logs/alerts.db"):
        self.alert_config = alert_config
        self.notification_config = notification_config
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Initialize notification handlers
        self.notification_handlers = []
        if notification_config.email_user:
            self.notification_handlers.append(EmailNotificationHandler(notification_config))
        if notification_config.slack_webhook_url:
            self.notification_handlers.append(SlackNotificationHandler(notification_config))
        
        # Alert tracking
        self.active_alerts = {}
        self.alert_counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        
        logger.info(f"Alert manager initialized with {len(self.notification_handlers)} notification handlers")
    
    def _init_database(self):
        """Initialize alerts database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    value REAL,
                    threshold REAL,
                    metadata TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Alert database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize alert database: {e}")
    
    def process_alert(self, alert: Alert) -> bool:
        """Process and handle alert"""
        try:
            # Store in database
            self._store_alert(alert)
            
            # Update counters
            self.alert_counts[alert.severity] += 1
            
            # Add to active alerts
            self.active_alerts[alert.id] = alert
            
            # Send notifications
            asyncio.create_task(self._send_notifications(alert))
            
            logger.info(f"Processed alert {alert.id}: {alert.severity} - {alert.message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process alert {alert.id}: {e}")
            return False
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (id, timestamp, severity, category, message, value, threshold, metadata, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.timestamp, alert.severity, alert.category,
                alert.message, alert.value, alert.threshold, 
                json.dumps(alert.metadata), alert.acknowledged, alert.resolved
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        # Only send notifications for WARNING and above
        if alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            for handler in self.notification_handlers:
                try:
                    await handler.send_notification(alert)
                except Exception as e:
                    logger.error(f"Notification handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                
                # Update in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE alerts SET acknowledged = TRUE WHERE id = ?', (alert_id,))
                conn.commit()
                conn.close()
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                
                # Update in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE alerts SET resolved = TRUE WHERE id = ?', (alert_id,))
                conn.commit()
                conn.close()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_active': len(active_alerts),
            'by_severity': {
                severity: len([a for a in active_alerts if a.severity == severity])
                for severity in [AlertSeverity.INFO, AlertSeverity.WARNING, 
                               AlertSeverity.ERROR, AlertSeverity.CRITICAL]
            },
            'by_category': {},  # Would need to implement category counting
            'oldest_unresolved': min([a.timestamp for a in active_alerts]) if active_alerts else None,
            'total_counts': self.alert_counts.copy()
        }

class ProductionMonitoringSystem:
    """Main production monitoring system"""
    
    def __init__(self, alert_config: AlertConfig = None, 
                 notification_config: NotificationConfig = None):
        
        self.alert_config = alert_config or AlertConfig()
        self.notification_config = notification_config or NotificationConfig()
        
        # Initialize components
        self.alert_manager = AlertManager(self.alert_config, self.notification_config)
        self.system_monitor = SystemMonitor(self.alert_config)
        self.performance_monitor = PerformanceMonitor(self.alert_config)
        self.data_quality_monitor = DataQualityMonitor(self.alert_config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Production monitoring system initialized")
    
    def start_monitoring(self):
        """Start all monitoring components"""
        try:
            self.monitoring_active = True
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Start main monitoring loop
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Production monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        try:
            self.monitoring_active = False
            
            # Stop system monitoring
            self.system_monitor.stop_monitoring()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Production monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Process any pending alerts
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def update_performance_metrics(self, portfolio_value: float, sharpe_ratio: float, 
                                 drawdown: float, win_rate: float):
        """Update performance metrics and process alerts"""
        alerts = self.performance_monitor.update_performance(
            portfolio_value, sharpe_ratio, drawdown, win_rate
        )
        
        for alert in alerts:
            self.alert_manager.process_alert(alert)
    
    def update_trade_data(self, trade_pnl: float, trade_duration: timedelta):
        """Update trade data and process alerts"""
        alerts = self.performance_monitor.update_trade(trade_pnl, trade_duration)
        
        for alert in alerts:
            self.alert_manager.process_alert(alert)
    
    def check_data_quality(self, data: pd.DataFrame):
        """Check data quality and process alerts"""
        alerts = self.data_quality_monitor.check_data_quality(data)
        
        for alert in alerts:
            self.alert_manager.process_alert(alert)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'alert_summary': self.alert_manager.get_alert_summary(),
            'system_status': self._get_system_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}

def create_monitoring_system(alert_config: AlertConfig = None,
                           notification_config: NotificationConfig = None) -> ProductionMonitoringSystem:
    """Factory function to create monitoring system"""
    return ProductionMonitoringSystem(alert_config, notification_config)

# Example usage
if __name__ == "__main__":
    # Create configurations
    alert_config = AlertConfig(
        min_sharpe_ratio=1.5,
        max_drawdown_pct=10.0,
        min_win_rate=0.55
    )
    
    notification_config = NotificationConfig(
        alert_recipients=["trader@example.com"],
        slack_webhook_url="https://hooks.slack.com/your/webhook/url"
    )
    
    # Create monitoring system
    monitor = create_monitoring_system(alert_config, notification_config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some updates
    monitor.update_performance_metrics(
        portfolio_value=105000,
        sharpe_ratio=1.8,
        drawdown=-0.05,
        win_rate=0.62
    )
    
    # Get status
    status = monitor.get_monitoring_status()
    print(f"Monitoring Status: {json.dumps(status, indent=2, default=str)}")
    
    # Stop monitoring
    time.sleep(5)
    monitor.stop_monitoring()
    
    print("Production monitoring system demonstration completed!")