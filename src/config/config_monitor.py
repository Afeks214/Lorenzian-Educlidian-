"""
Configuration Monitoring and Alerting System
Real-time monitoring, alerting, and compliance checking for configurations.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import psutil
import os


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


@dataclass
class ConfigAlert:
    """Configuration alert"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    config_name: str
    field_path: Optional[str]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    id: str
    name: str
    description: str
    category: str
    severity: AlertSeverity
    config_path: str
    validation_func: str
    parameters: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ConfigMetric:
    """Configuration metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class MonitoringStats:
    """Monitoring statistics"""
    total_configs: int
    monitored_configs: int
    active_alerts: int
    compliance_violations: int
    last_check: datetime
    uptime: float
    memory_usage: float
    cpu_usage: float


class ConfigMonitor:
    """
    Configuration monitoring system with:
    - Real-time configuration monitoring
    - Alert management
    - Compliance checking
    - Performance metrics
    - Health monitoring
    """

    def __init__(self, config_manager, check_interval: int = 60):
        self.config_manager = config_manager
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Storage paths
        self.monitoring_path = Path(__file__).parent.parent.parent / "monitoring"
        self.alerts_path = self.monitoring_path / "alerts"
        self.compliance_path = self.monitoring_path / "compliance"
        self.metrics_path = self.monitoring_path / "metrics"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # State
        self.alerts: Dict[str, ConfigAlert] = {}
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.metrics: List[ConfigMetric] = []
        self.config_checksums: Dict[str, str] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[ConfigAlert], None]] = []
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time = datetime.now()
        
        # Load state
        self._load_alerts()
        self._load_compliance_rules()
        self._load_default_compliance_rules()
        
        # Start monitoring
        self.start_monitoring()
        
        self.logger.info("ConfigMonitor initialized")

    def _ensure_directories(self):
        """Ensure all monitoring directories exist"""
        for path in [self.monitoring_path, self.alerts_path, self.compliance_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_alerts(self):
        """Load alerts from disk"""
        alerts_file = self.alerts_path / "alerts.json"
        
        if alerts_file.exists():
            try:
                with open(alerts_file, 'r') as f:
                    data = json.load(f)
                
                for alert_id, alert_data in data.items():
                    # Convert datetime strings back to datetime objects
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    if alert_data.get('resolved_at'):
                        alert_data['resolved_at'] = datetime.fromisoformat(alert_data['resolved_at'])
                    
                    # Convert enum string back to enum
                    alert_data['severity'] = AlertSeverity(alert_data['severity'])
                    
                    self.alerts[alert_id] = ConfigAlert(**alert_data)
                
                self.logger.info(f"Loaded {len(self.alerts)} alerts")
                
            except Exception as e:
                self.logger.error(f"Failed to load alerts: {e}")

    def _save_alerts(self):
        """Save alerts to disk"""
        alerts_file = self.alerts_path / "alerts.json"
        
        # Convert to serializable format
        data = {}
        for alert_id, alert in self.alerts.items():
            alert_dict = asdict(alert)
            
            # Convert datetime objects to ISO strings
            alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            if alert_dict.get('resolved_at'):
                alert_dict['resolved_at'] = alert_dict['resolved_at'].isoformat()
            
            # Convert enum to string
            alert_dict['severity'] = alert_dict['severity'].value
            
            data[alert_id] = alert_dict
        
        with open(alerts_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_compliance_rules(self):
        """Load compliance rules from disk"""
        rules_file = self.compliance_path / "rules.json"
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    data = json.load(f)
                
                for rule_id, rule_data in data.items():
                    # Convert datetime strings back to datetime objects
                    rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                    rule_data['updated_at'] = datetime.fromisoformat(rule_data['updated_at'])
                    
                    # Convert enum string back to enum
                    rule_data['severity'] = AlertSeverity(rule_data['severity'])
                    
                    self.compliance_rules[rule_id] = ComplianceRule(**rule_data)
                
                self.logger.info(f"Loaded {len(self.compliance_rules)} compliance rules")
                
            except Exception as e:
                self.logger.error(f"Failed to load compliance rules: {e}")

    def _save_compliance_rules(self):
        """Save compliance rules to disk"""
        rules_file = self.compliance_path / "rules.json"
        
        # Convert to serializable format
        data = {}
        for rule_id, rule in self.compliance_rules.items():
            rule_dict = asdict(rule)
            
            # Convert datetime objects to ISO strings
            rule_dict['created_at'] = rule_dict['created_at'].isoformat()
            rule_dict['updated_at'] = rule_dict['updated_at'].isoformat()
            
            # Convert enum to string
            rule_dict['severity'] = rule_dict['severity'].value
            
            data[rule_id] = rule_dict
        
        with open(rules_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_default_compliance_rules(self):
        """Load default compliance rules"""
        if not self.compliance_rules:
            default_rules = [
                ComplianceRule(
                    id="password_strength",
                    name="Password Strength",
                    description="Ensure passwords meet minimum security requirements",
                    category="security",
                    severity=AlertSeverity.ERROR,
                    config_path="*.password",
                    validation_func="validate_password_strength",
                    parameters={"min_length": 12, "require_special": True}
                ),
                ComplianceRule(
                    id="api_key_format",
                    name="API Key Format",
                    description="Ensure API keys follow proper format",
                    category="security",
                    severity=AlertSeverity.WARNING,
                    config_path="*.api_key",
                    validation_func="validate_api_key_format",
                    parameters={"min_length": 32}
                ),
                ComplianceRule(
                    id="max_position_size",
                    name="Maximum Position Size",
                    description="Ensure position sizes are within acceptable limits",
                    category="risk",
                    severity=AlertSeverity.CRITICAL,
                    config_path="risk_management.max_position_size",
                    validation_func="validate_position_size",
                    parameters={"max_value": 1000000}
                ),
                ComplianceRule(
                    id="ssl_enabled",
                    name="SSL Enabled",
                    description="Ensure SSL is enabled for production",
                    category="security",
                    severity=AlertSeverity.CRITICAL,
                    config_path="system.ssl_enabled",
                    validation_func="validate_ssl_enabled",
                    parameters={"required_for_prod": True}
                ),
                ComplianceRule(
                    id="backup_enabled",
                    name="Backup Enabled",
                    description="Ensure backups are enabled",
                    category="data_protection",
                    severity=AlertSeverity.WARNING,
                    config_path="system.backup_enabled",
                    validation_func="validate_backup_enabled",
                    parameters={"required": True}
                )
            ]
            
            for rule in default_rules:
                self.compliance_rules[rule.id] = rule
            
            self._save_compliance_rules()

    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self._stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Configuration monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if self.monitoring_thread is not None:
            self._stop_event.set()
            self.monitoring_thread.join(timeout=10)
            self.logger.info("Configuration monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Check configuration changes
                self._check_configuration_changes()
                
                # Check compliance
                self._check_compliance()
                
                # Collect metrics
                self._collect_metrics()
                
                # Save state
                self._save_alerts()
                
                # Wait for next check
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _check_configuration_changes(self):
        """Check for configuration changes"""
        with self._lock:
            current_configs = self.config_manager.get_all_configs()
            
            for config_name, config_data in current_configs.items():
                # Calculate checksum
                content = json.dumps(config_data, sort_keys=True)
                checksum = hashlib.md5(content.encode()).hexdigest()
                
                # Check if changed
                if config_name in self.config_checksums:
                    if self.config_checksums[config_name] != checksum:
                        self._create_alert(
                            severity=AlertSeverity.INFO,
                            title="Configuration Changed",
                            message=f"Configuration '{config_name}' has been modified",
                            config_name=config_name
                        )
                
                # Update checksum
                self.config_checksums[config_name] = checksum

    def _check_compliance(self):
        """Check compliance rules"""
        with self._lock:
            for rule_id, rule in self.compliance_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Get configuration value
                    config_value = self._get_config_value(rule.config_path)
                    
                    # Validate
                    is_compliant = self._validate_compliance_rule(rule, config_value)
                    
                    if not is_compliant:
                        self._create_alert(
                            severity=rule.severity,
                            title=f"Compliance Violation: {rule.name}",
                            message=rule.description,
                            config_name=rule.config_path.split('.')[0],
                            field_path=rule.config_path,
                            metadata={"rule_id": rule_id, "category": rule.category}
                        )
                
                except Exception as e:
                    self.logger.error(f"Error checking compliance rule {rule_id}: {e}")

    def _get_config_value(self, config_path: str) -> Any:
        """Get configuration value by path"""
        parts = config_path.split('.')
        config_name = parts[0]
        
        config_data = self.config_manager.get_config(config_name)
        
        if len(parts) == 1:
            return config_data
        
        # Navigate to nested value
        value = config_data
        for part in parts[1:]:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value

    def _validate_compliance_rule(self, rule: ComplianceRule, config_value: Any) -> bool:
        """Validate a compliance rule"""
        validation_func = rule.validation_func
        parameters = rule.parameters
        
        # Built-in validation functions
        if validation_func == "validate_password_strength":
            return self._validate_password_strength(config_value, parameters)
        elif validation_func == "validate_api_key_format":
            return self._validate_api_key_format(config_value, parameters)
        elif validation_func == "validate_position_size":
            return self._validate_position_size(config_value, parameters)
        elif validation_func == "validate_ssl_enabled":
            return self._validate_ssl_enabled(config_value, parameters)
        elif validation_func == "validate_backup_enabled":
            return self._validate_backup_enabled(config_value, parameters)
        else:
            self.logger.warning(f"Unknown validation function: {validation_func}")
            return True

    def _validate_password_strength(self, password: str, params: Dict[str, Any]) -> bool:
        """Validate password strength"""
        if not isinstance(password, str):
            return False
        
        if len(password) < params.get("min_length", 8):
            return False
        
        if params.get("require_special", False):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True

    def _validate_api_key_format(self, api_key: str, params: Dict[str, Any]) -> bool:
        """Validate API key format"""
        if not isinstance(api_key, str):
            return False
        
        if len(api_key) < params.get("min_length", 32):
            return False
        
        # Check for obvious test values
        test_values = ["test", "demo", "example", "placeholder"]
        if api_key.lower() in test_values:
            return False
        
        return True

    def _validate_position_size(self, position_size: Any, params: Dict[str, Any]) -> bool:
        """Validate position size"""
        if not isinstance(position_size, (int, float)):
            return False
        
        max_value = params.get("max_value", 1000000)
        if position_size > max_value:
            return False
        
        return True

    def _validate_ssl_enabled(self, ssl_enabled: Any, params: Dict[str, Any]) -> bool:
        """Validate SSL enabled"""
        if params.get("required_for_prod", False):
            environment = os.getenv("ENVIRONMENT", "development")
            if environment == "production" and not ssl_enabled:
                return False
        
        return True

    def _validate_backup_enabled(self, backup_enabled: Any, params: Dict[str, Any]) -> bool:
        """Validate backup enabled"""
        if params.get("required", False) and not backup_enabled:
            return False
        
        return True

    def _collect_metrics(self):
        """Collect system metrics"""
        now = datetime.now()
        
        # System metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        self.metrics.extend([
            ConfigMetric("system.memory_usage", memory_usage, "percent", now),
            ConfigMetric("system.cpu_usage", cpu_usage, "percent", now),
            ConfigMetric("monitor.active_alerts", len([a for a in self.alerts.values() if not a.resolved]), "count", now),
            ConfigMetric("monitor.compliance_rules", len(self.compliance_rules), "count", now),
            ConfigMetric("monitor.configs_monitored", len(self.config_checksums), "count", now),
        ])
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    def _create_alert(self, severity: AlertSeverity, title: str, message: str,
                     config_name: str, field_path: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        alert_id = hashlib.md5(f"{title}:{config_name}:{field_path}".encode()).hexdigest()
        
        # Check if alert already exists and is unresolved
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return alert_id
        
        alert = ConfigAlert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            config_name=config_name,
            field_path=field_path,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.info(f"Created alert: {title} ({severity.value})")
        return alert_id

    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            if alert.resolved:
                return False
            
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolved_by = resolved_by
            
            self._save_alerts()
            
            self.logger.info(f"Resolved alert: {alert.title}")
            return True

    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                  resolved: Optional[bool] = None,
                  config_name: Optional[str] = None,
                  hours_back: int = 24) -> List[ConfigAlert]:
        """Get alerts with filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_alerts = []
        for alert in self.alerts.values():
            if alert.timestamp < cutoff_time:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            if resolved is not None and alert.resolved != resolved:
                continue
            
            if config_name and alert.config_name != config_name:
                continue
            
            filtered_alerts.append(alert)
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    def get_compliance_status(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance status"""
        compliant_rules = 0
        non_compliant_rules = 0
        
        for rule in self.compliance_rules.values():
            if not rule.enabled:
                continue
            
            if config_name and not rule.config_path.startswith(config_name):
                continue
            
            try:
                config_value = self._get_config_value(rule.config_path)
                is_compliant = self._validate_compliance_rule(rule, config_value)
                
                if is_compliant:
                    compliant_rules += 1
                else:
                    non_compliant_rules += 1
            except Exception:
                pass
        
        total_rules = compliant_rules + non_compliant_rules
        compliance_percentage = (compliant_rules / total_rules * 100) if total_rules > 0 else 0
        
        return {
            'compliant_rules': compliant_rules,
            'non_compliant_rules': non_compliant_rules,
            'total_rules': total_rules,
            'compliance_percentage': compliance_percentage,
            'status': ComplianceStatus.COMPLIANT if non_compliant_rules == 0 else ComplianceStatus.NON_COMPLIANT
        }

    def add_compliance_rule(self, rule: ComplianceRule) -> bool:
        """Add a compliance rule"""
        with self._lock:
            if rule.id in self.compliance_rules:
                return False
            
            self.compliance_rules[rule.id] = rule
            self._save_compliance_rules()
            
            self.logger.info(f"Added compliance rule: {rule.name}")
            return True

    def update_compliance_rule(self, rule_id: str, **kwargs) -> bool:
        """Update a compliance rule"""
        with self._lock:
            if rule_id not in self.compliance_rules:
                return False
            
            rule = self.compliance_rules[rule_id]
            
            for field, value in kwargs.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            rule.updated_at = datetime.now()
            self._save_compliance_rules()
            
            self.logger.info(f"Updated compliance rule: {rule.name}")
            return True

    def delete_compliance_rule(self, rule_id: str) -> bool:
        """Delete a compliance rule"""
        with self._lock:
            if rule_id not in self.compliance_rules:
                return False
            
            del self.compliance_rules[rule_id]
            self._save_compliance_rules()
            
            self.logger.info(f"Deleted compliance rule: {rule_id}")
            return True

    def add_alert_handler(self, handler: Callable[[ConfigAlert], None]):
        """Add an alert handler"""
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[ConfigAlert], None]):
        """Remove an alert handler"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

    def get_metrics(self, metric_name: Optional[str] = None, 
                   hours_back: int = 24) -> List[ConfigMetric]:
        """Get metrics with filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_metrics = []
        for metric in self.metrics:
            if metric.timestamp < cutoff_time:
                continue
            
            if metric_name and metric.name != metric_name:
                continue
            
            filtered_metrics.append(metric)
        
        return sorted(filtered_metrics, key=lambda x: x.timestamp, reverse=True)

    def get_monitoring_stats(self) -> MonitoringStats:
        """Get monitoring statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return MonitoringStats(
            total_configs=len(self.config_manager.get_all_configs()),
            monitored_configs=len(self.config_checksums),
            active_alerts=len([a for a in self.alerts.values() if not a.resolved]),
            compliance_violations=self.get_compliance_status()['non_compliant_rules'],
            last_check=datetime.now(),
            uptime=uptime,
            memory_usage=psutil.virtual_memory().percent,
            cpu_usage=psutil.cpu_percent()
        )

    def create_monitoring_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Create a comprehensive monitoring report"""
        stats = self.get_monitoring_stats()
        alerts = self.get_alerts(hours_back=hours_back)
        compliance = self.get_compliance_status()
        
        # Alert summary
        alert_summary = {}
        for severity in AlertSeverity:
            alert_summary[severity.value] = len([a for a in alerts if a.severity == severity])
        
        # Top configurations by alert count
        config_alerts = {}
        for alert in alerts:
            config_alerts[alert.config_name] = config_alerts.get(alert.config_name, 0) + 1
        
        top_configs = sorted(config_alerts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'time_range': {
                'start': (datetime.now() - timedelta(hours=hours_back)).isoformat(),
                'end': datetime.now().isoformat()
            },
            'statistics': asdict(stats),
            'compliance_status': compliance,
            'alert_summary': alert_summary,
            'top_configs_by_alerts': top_configs,
            'recent_alerts': [asdict(alert) for alert in alerts[:10]],
            'system_health': {
                'monitoring_active': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                'alert_handlers': len(self.alert_handlers),
                'compliance_rules': len(self.compliance_rules)
            }
        }

    def export_monitoring_data(self, export_path: Path):
        """Export monitoring data"""
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export alerts
        alerts_data = {}
        for alert_id, alert in self.alerts.items():
            alert_dict = asdict(alert)
            alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            if alert_dict.get('resolved_at'):
                alert_dict['resolved_at'] = alert_dict['resolved_at'].isoformat()
            alert_dict['severity'] = alert_dict['severity'].value
            alerts_data[alert_id] = alert_dict
        
        with open(export_path / "alerts.json", 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        # Export compliance rules
        rules_data = {}
        for rule_id, rule in self.compliance_rules.items():
            rule_dict = asdict(rule)
            rule_dict['created_at'] = rule_dict['created_at'].isoformat()
            rule_dict['updated_at'] = rule_dict['updated_at'].isoformat()
            rule_dict['severity'] = rule_dict['severity'].value
            rules_data[rule_id] = rule_dict
        
        with open(export_path / "compliance_rules.json", 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        # Export metrics
        metrics_data = []
        for metric in self.metrics:
            metric_dict = asdict(metric)
            metric_dict['timestamp'] = metric_dict['timestamp'].isoformat()
            metrics_data.append(metric_dict)
        
        with open(export_path / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Export report
        report = self.create_monitoring_report()
        with open(export_path / "monitoring_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring data exported to {export_path}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            'monitoring_active': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'disk_usage': psutil.disk_usage('/').percent,
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
            'compliance_violations': self.get_compliance_status()['non_compliant_rules'],
            'last_check': datetime.now().isoformat()
        }