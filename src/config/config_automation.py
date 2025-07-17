"""
Configuration Automation System
Automated configuration deployment, drift detection, backup/recovery, and compliance.
"""

import json
import yaml
import logging
import threading
import subprocess
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import tempfile
import os
import git


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DriftStatus(Enum):
    """Configuration drift status"""
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    DRIFT_CORRECTED = "drift_corrected"


@dataclass
class DeploymentJob:
    """Configuration deployment job"""
    id: str
    name: str
    description: str
    config_changes: Dict[str, Any]
    target_environment: str
    status: DeploymentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DriftDetection:
    """Configuration drift detection result"""
    id: str
    config_name: str
    expected_checksum: str
    actual_checksum: str
    drift_detected: bool
    drift_details: Dict[str, Any]
    timestamp: datetime
    corrected: bool = False
    corrected_at: Optional[datetime] = None


@dataclass
class BackupRecord:
    """Configuration backup record"""
    id: str
    name: str
    description: str
    configs: List[str]
    backup_path: Path
    created_at: datetime
    size_bytes: int
    retention_days: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AutomationRule:
    """Automation rule definition"""
    id: str
    name: str
    description: str
    trigger_type: str  # 'schedule', 'event', 'drift'
    trigger_config: Dict[str, Any]
    actions: List[str]
    enabled: bool = True
    created_at: datetime = None
    last_executed: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ConfigAutomation:
    """
    Configuration automation system with:
    - Automated deployment
    - Drift detection and correction
    - Backup and recovery
    - Compliance automation
    - CI/CD integration
    """

    def __init__(self, config_manager, backup_path: Optional[Path] = None):
        self.config_manager = config_manager
        self.backup_path = backup_path or Path(__file__).parent.parent.parent / "backups"
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Storage paths
        self.automation_path = Path(__file__).parent.parent.parent / "automation"
        self.deployments_path = self.automation_path / "deployments"
        self.drift_path = self.automation_path / "drift"
        self.rules_path = self.automation_path / "rules"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # State
        self.deployment_jobs: Dict[str, DeploymentJob] = {}
        self.drift_detections: Dict[str, DriftDetection] = {}
        self.backup_records: Dict[str, BackupRecord] = {}
        self.automation_rules: Dict[str, AutomationRule] = {}
        
        # Baseline checksums for drift detection
        self.baseline_checksums: Dict[str, str] = {}
        
        # Event handlers
        self.deployment_handlers: List[Callable[[DeploymentJob], None]] = []
        self.drift_handlers: List[Callable[[DriftDetection], None]] = []
        
        # Automation thread
        self.automation_thread: Optional[threading.Thread] = None
        
        # Load state
        self._load_deployment_jobs()
        self._load_drift_detections()
        self._load_backup_records()
        self._load_automation_rules()
        self._load_default_automation_rules()
        
        # Initialize baselines
        self._initialize_baselines()
        
        # Start automation
        self.start_automation()
        
        self.logger.info("ConfigAutomation initialized")

    def _ensure_directories(self):
        """Ensure all automation directories exist"""
        for path in [self.automation_path, self.deployments_path, self.drift_path, 
                    self.rules_path, self.backup_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_deployment_jobs(self):
        """Load deployment jobs from disk"""
        jobs_file = self.deployments_path / "jobs.json"
        
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    data = json.load(f)
                
                for job_id, job_data in data.items():
                    # Convert datetime strings back to datetime objects
                    job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                    if job_data.get('started_at'):
                        job_data['started_at'] = datetime.fromisoformat(job_data['started_at'])
                    if job_data.get('completed_at'):
                        job_data['completed_at'] = datetime.fromisoformat(job_data['completed_at'])
                    
                    # Convert enum string back to enum
                    job_data['status'] = DeploymentStatus(job_data['status'])
                    
                    self.deployment_jobs[job_id] = DeploymentJob(**job_data)
                
                self.logger.info(f"Loaded {len(self.deployment_jobs)} deployment jobs")
                
            except Exception as e:
                self.logger.error(f"Failed to load deployment jobs: {e}")

    def _save_deployment_jobs(self):
        """Save deployment jobs to disk"""
        jobs_file = self.deployments_path / "jobs.json"
        
        # Convert to serializable format
        data = {}
        for job_id, job in self.deployment_jobs.items():
            job_dict = asdict(job)
            
            # Convert datetime objects to ISO strings
            job_dict['created_at'] = job_dict['created_at'].isoformat()
            if job_dict.get('started_at'):
                job_dict['started_at'] = job_dict['started_at'].isoformat()
            if job_dict.get('completed_at'):
                job_dict['completed_at'] = job_dict['completed_at'].isoformat()
            
            # Convert enum to string
            job_dict['status'] = job_dict['status'].value
            
            data[job_id] = job_dict
        
        with open(jobs_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_drift_detections(self):
        """Load drift detections from disk"""
        drift_file = self.drift_path / "detections.json"
        
        if drift_file.exists():
            try:
                with open(drift_file, 'r') as f:
                    data = json.load(f)
                
                for detection_id, detection_data in data.items():
                    # Convert datetime strings back to datetime objects
                    detection_data['timestamp'] = datetime.fromisoformat(detection_data['timestamp'])
                    if detection_data.get('corrected_at'):
                        detection_data['corrected_at'] = datetime.fromisoformat(detection_data['corrected_at'])
                    
                    self.drift_detections[detection_id] = DriftDetection(**detection_data)
                
                self.logger.info(f"Loaded {len(self.drift_detections)} drift detections")
                
            except Exception as e:
                self.logger.error(f"Failed to load drift detections: {e}")

    def _save_drift_detections(self):
        """Save drift detections to disk"""
        drift_file = self.drift_path / "detections.json"
        
        # Convert to serializable format
        data = {}
        for detection_id, detection in self.drift_detections.items():
            detection_dict = asdict(detection)
            
            # Convert datetime objects to ISO strings
            detection_dict['timestamp'] = detection_dict['timestamp'].isoformat()
            if detection_dict.get('corrected_at'):
                detection_dict['corrected_at'] = detection_dict['corrected_at'].isoformat()
            
            data[detection_id] = detection_dict
        
        with open(drift_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_backup_records(self):
        """Load backup records from disk"""
        backup_file = self.backup_path / "records.json"
        
        if backup_file.exists():
            try:
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                
                for record_id, record_data in data.items():
                    # Convert datetime strings back to datetime objects
                    record_data['created_at'] = datetime.fromisoformat(record_data['created_at'])
                    
                    # Convert path strings back to Path objects
                    record_data['backup_path'] = Path(record_data['backup_path'])
                    
                    self.backup_records[record_id] = BackupRecord(**record_data)
                
                self.logger.info(f"Loaded {len(self.backup_records)} backup records")
                
            except Exception as e:
                self.logger.error(f"Failed to load backup records: {e}")

    def _save_backup_records(self):
        """Save backup records to disk"""
        backup_file = self.backup_path / "records.json"
        
        # Convert to serializable format
        data = {}
        for record_id, record in self.backup_records.items():
            record_dict = asdict(record)
            
            # Convert datetime objects to ISO strings
            record_dict['created_at'] = record_dict['created_at'].isoformat()
            
            # Convert Path objects to strings
            record_dict['backup_path'] = str(record_dict['backup_path'])
            
            data[record_id] = record_dict
        
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_automation_rules(self):
        """Load automation rules from disk"""
        rules_file = self.rules_path / "rules.json"
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    data = json.load(f)
                
                for rule_id, rule_data in data.items():
                    # Convert datetime strings back to datetime objects
                    rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                    if rule_data.get('last_executed'):
                        rule_data['last_executed'] = datetime.fromisoformat(rule_data['last_executed'])
                    
                    self.automation_rules[rule_id] = AutomationRule(**rule_data)
                
                self.logger.info(f"Loaded {len(self.automation_rules)} automation rules")
                
            except Exception as e:
                self.logger.error(f"Failed to load automation rules: {e}")

    def _save_automation_rules(self):
        """Save automation rules to disk"""
        rules_file = self.rules_path / "rules.json"
        
        # Convert to serializable format
        data = {}
        for rule_id, rule in self.automation_rules.items():
            rule_dict = asdict(rule)
            
            # Convert datetime objects to ISO strings
            rule_dict['created_at'] = rule_dict['created_at'].isoformat()
            if rule_dict.get('last_executed'):
                rule_dict['last_executed'] = rule_dict['last_executed'].isoformat()
            
            data[rule_id] = rule_dict
        
        with open(rules_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_default_automation_rules(self):
        """Load default automation rules"""
        if not self.automation_rules:
            default_rules = [
                AutomationRule(
                    id="daily_backup",
                    name="Daily Configuration Backup",
                    description="Create daily backup of all configurations",
                    trigger_type="schedule",
                    trigger_config={"cron": "0 2 * * *"},  # 2 AM daily
                    actions=["backup_configs"]
                ),
                AutomationRule(
                    id="drift_correction",
                    name="Automatic Drift Correction",
                    description="Automatically correct configuration drift",
                    trigger_type="drift",
                    trigger_config={"auto_correct": True},
                    actions=["correct_drift"]
                ),
                AutomationRule(
                    id="cleanup_old_backups",
                    name="Cleanup Old Backups",
                    description="Remove backups older than retention period",
                    trigger_type="schedule",
                    trigger_config={"cron": "0 3 * * 0"},  # 3 AM every Sunday
                    actions=["cleanup_backups"]
                ),
                AutomationRule(
                    id="compliance_check",
                    name="Compliance Check",
                    description="Regular compliance checking",
                    trigger_type="schedule",
                    trigger_config={"cron": "0 */6 * * *"},  # Every 6 hours
                    actions=["check_compliance"]
                )
            ]
            
            for rule in default_rules:
                self.automation_rules[rule.id] = rule
            
            self._save_automation_rules()

    def _initialize_baselines(self):
        """Initialize baseline checksums"""
        current_configs = self.config_manager.get_all_configs()
        
        for config_name, config_data in current_configs.items():
            content = json.dumps(config_data, sort_keys=True)
            checksum = hashlib.md5(content.encode()).hexdigest()
            self.baseline_checksums[config_name] = checksum

    def start_automation(self):
        """Start the automation thread"""
        if self.automation_thread is None or not self.automation_thread.is_alive():
            self._stop_event.clear()
            self.automation_thread = threading.Thread(target=self._automation_loop, daemon=True)
            self.automation_thread.start()
            self.logger.info("Configuration automation started")

    def stop_automation(self):
        """Stop the automation thread"""
        if self.automation_thread is not None:
            self._stop_event.set()
            self.automation_thread.join(timeout=10)
            self.logger.info("Configuration automation stopped")

    def _automation_loop(self):
        """Main automation loop"""
        while not self._stop_event.is_set():
            try:
                # Process pending deployments
                self._process_deployments()
                
                # Check for drift
                self._check_drift()
                
                # Process automation rules
                self._process_automation_rules()
                
                # Cleanup old records
                self._cleanup_old_records()
                
                # Wait before next check
                self._stop_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in automation loop: {e}")

    def _process_deployments(self):
        """Process pending deployment jobs"""
        with self._lock:
            pending_jobs = [job for job in self.deployment_jobs.values() 
                          if job.status == DeploymentStatus.PENDING]
            
            for job in pending_jobs:
                self._execute_deployment(job)

    def _execute_deployment(self, job: DeploymentJob):
        """Execute a deployment job"""
        try:
            job.status = DeploymentStatus.RUNNING
            job.started_at = datetime.now()
            self._save_deployment_jobs()
            
            # Create rollback data
            rollback_data = {}
            for config_name in job.config_changes.keys():
                rollback_data[config_name] = self.config_manager.get_config(config_name)
            
            job.rollback_data = rollback_data
            
            # Apply configuration changes
            for config_name, config_data in job.config_changes.items():
                self.config_manager.set_config(config_name, config_data, 
                                             user="automation", 
                                             reason=f"Deployment: {job.name}")
            
            # Validate deployment
            if self._validate_deployment(job):
                job.status = DeploymentStatus.SUCCESS
                job.completed_at = datetime.now()
                self.logger.info(f"Deployment '{job.name}' completed successfully")
            else:
                raise Exception("Deployment validation failed")
            
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self.logger.error(f"Deployment '{job.name}' failed: {e}")
            
            # Attempt rollback
            if job.rollback_data:
                self._rollback_deployment(job)
        
        finally:
            self._save_deployment_jobs()
            
            # Notify handlers
            for handler in self.deployment_handlers:
                try:
                    handler(job)
                except Exception as e:
                    self.logger.error(f"Error in deployment handler: {e}")

    def _validate_deployment(self, job: DeploymentJob) -> bool:
        """Validate a deployment"""
        try:
            # Check if all configurations were applied
            for config_name, expected_config in job.config_changes.items():
                current_config = self.config_manager.get_config(config_name)
                if current_config != expected_config:
                    return False
            
            # Additional validation can be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment validation error: {e}")
            return False

    def _rollback_deployment(self, job: DeploymentJob):
        """Rollback a failed deployment"""
        try:
            if not job.rollback_data:
                return
            
            for config_name, rollback_config in job.rollback_data.items():
                self.config_manager.set_config(config_name, rollback_config,
                                             user="automation",
                                             reason=f"Rollback: {job.name}")
            
            job.status = DeploymentStatus.ROLLED_BACK
            self.logger.info(f"Rolled back deployment '{job.name}'")
            
        except Exception as e:
            self.logger.error(f"Rollback failed for deployment '{job.name}': {e}")

    def _check_drift(self):
        """Check for configuration drift"""
        with self._lock:
            current_configs = self.config_manager.get_all_configs()
            
            for config_name, config_data in current_configs.items():
                content = json.dumps(config_data, sort_keys=True)
                current_checksum = hashlib.md5(content.encode()).hexdigest()
                
                expected_checksum = self.baseline_checksums.get(config_name)
                
                if expected_checksum and current_checksum != expected_checksum:
                    self._detect_drift(config_name, expected_checksum, current_checksum, config_data)

    def _detect_drift(self, config_name: str, expected_checksum: str, 
                     actual_checksum: str, config_data: Dict[str, Any]):
        """Detect and record configuration drift"""
        detection_id = f"{config_name}_{actual_checksum}"
        
        if detection_id in self.drift_detections:
            return  # Already detected
        
        # Calculate drift details
        drift_details = {
            "config_name": config_name,
            "checksum_mismatch": True,
            "detected_at": datetime.now().isoformat()
        }
        
        detection = DriftDetection(
            id=detection_id,
            config_name=config_name,
            expected_checksum=expected_checksum,
            actual_checksum=actual_checksum,
            drift_detected=True,
            drift_details=drift_details,
            timestamp=datetime.now()
        )
        
        self.drift_detections[detection_id] = detection
        self._save_drift_detections()
        
        # Notify handlers
        for handler in self.drift_handlers:
            try:
                handler(detection)
            except Exception as e:
                self.logger.error(f"Error in drift handler: {e}")
        
        self.logger.warning(f"Configuration drift detected for '{config_name}'")

    def _process_automation_rules(self):
        """Process automation rules"""
        now = datetime.now()
        
        for rule in self.automation_rules.values():
            if not rule.enabled:
                continue
            
            if self._should_execute_rule(rule, now):
                self._execute_automation_rule(rule)

    def _should_execute_rule(self, rule: AutomationRule, now: datetime) -> bool:
        """Check if automation rule should be executed"""
        if rule.trigger_type == "schedule":
            # Simple schedule check (would need full cron parsing for production)
            if rule.last_executed is None:
                return True
            
            # Check if enough time has passed
            time_diff = now - rule.last_executed
            if time_diff.total_seconds() > 3600:  # 1 hour minimum
                return True
        
        elif rule.trigger_type == "drift":
            # Check if there are unresolved drift detections
            unresolved_drifts = [d for d in self.drift_detections.values() 
                               if d.drift_detected and not d.corrected]
            return len(unresolved_drifts) > 0
        
        return False

    def _execute_automation_rule(self, rule: AutomationRule):
        """Execute an automation rule"""
        try:
            for action in rule.actions:
                self._execute_action(action, rule)
            
            rule.last_executed = datetime.now()
            self._save_automation_rules()
            
            self.logger.info(f"Executed automation rule: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute automation rule '{rule.name}': {e}")

    def _execute_action(self, action: str, rule: AutomationRule):
        """Execute an automation action"""
        if action == "backup_configs":
            self.create_backup(f"automated_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        elif action == "correct_drift":
            self._correct_drift()
        
        elif action == "cleanup_backups":
            self._cleanup_old_backups()
        
        elif action == "check_compliance":
            # This would integrate with the monitoring system
            pass
        
        else:
            self.logger.warning(f"Unknown automation action: {action}")

    def _correct_drift(self):
        """Correct configuration drift"""
        for detection in self.drift_detections.values():
            if detection.drift_detected and not detection.corrected:
                # Reset to baseline (in production, this might be more sophisticated)
                self.baseline_checksums[detection.config_name] = detection.actual_checksum
                detection.corrected = True
                detection.corrected_at = datetime.now()
        
        self._save_drift_detections()

    def _cleanup_old_records(self):
        """Cleanup old records"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Cleanup old deployment jobs
        old_jobs = [job_id for job_id, job in self.deployment_jobs.items()
                   if job.created_at < cutoff_time]
        
        for job_id in old_jobs:
            del self.deployment_jobs[job_id]
        
        # Cleanup old drift detections
        old_detections = [detection_id for detection_id, detection in self.drift_detections.items()
                         if detection.timestamp < cutoff_time]
        
        for detection_id in old_detections:
            del self.drift_detections[detection_id]
        
        if old_jobs or old_detections:
            self._save_deployment_jobs()
            self._save_drift_detections()

    def _cleanup_old_backups(self):
        """Cleanup old backups"""
        for record in list(self.backup_records.values()):
            if record.retention_days > 0:
                expiry_date = record.created_at + timedelta(days=record.retention_days)
                if datetime.now() > expiry_date:
                    self._delete_backup(record.id)

    def create_deployment(self, name: str, description: str, config_changes: Dict[str, Any],
                         target_environment: str = "production") -> str:
        """Create a new deployment job"""
        job_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        job = DeploymentJob(
            id=job_id,
            name=name,
            description=description,
            config_changes=config_changes,
            target_environment=target_environment,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.deployment_jobs[job_id] = job
        self._save_deployment_jobs()
        
        self.logger.info(f"Created deployment job: {name}")
        return job_id

    def get_deployment_status(self, job_id: str) -> Optional[DeploymentJob]:
        """Get deployment status"""
        return self.deployment_jobs.get(job_id)

    def cancel_deployment(self, job_id: str) -> bool:
        """Cancel a pending deployment"""
        with self._lock:
            job = self.deployment_jobs.get(job_id)
            if job and job.status == DeploymentStatus.PENDING:
                job.status = DeploymentStatus.FAILED
                job.error_message = "Cancelled by user"
                job.completed_at = datetime.now()
                self._save_deployment_jobs()
                return True
            return False

    def create_backup(self, backup_name: str, description: str = "",
                     configs: Optional[List[str]] = None, 
                     retention_days: int = 30) -> str:
        """Create a configuration backup"""
        backup_id = hashlib.md5(f"{backup_name}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Get configurations to backup
        if configs is None:
            configs = list(self.config_manager.get_all_configs().keys())
        
        # Create backup directory
        backup_dir = self.backup_path / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup configurations
        total_size = 0
        for config_name in configs:
            config_data = self.config_manager.get_config(config_name)
            config_file = backup_dir / f"{config_name}.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            total_size += config_file.stat().st_size
        
        # Create metadata
        metadata = {
            'backup_id': backup_id,
            'backup_name': backup_name,
            'created_at': datetime.now().isoformat(),
            'configs': configs,
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        metadata_file = backup_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        total_size += metadata_file.stat().st_size
        
        # Create backup record
        record = BackupRecord(
            id=backup_id,
            name=backup_name,
            description=description,
            configs=configs,
            backup_path=backup_dir,
            created_at=datetime.now(),
            size_bytes=total_size,
            retention_days=retention_days
        )
        
        self.backup_records[backup_id] = record
        self._save_backup_records()
        
        self.logger.info(f"Created backup: {backup_name} ({len(configs)} configs)")
        return backup_id

    def restore_backup(self, backup_id: str, user: str = "automation") -> bool:
        """Restore configuration from backup"""
        with self._lock:
            record = self.backup_records.get(backup_id)
            if not record:
                self.logger.error(f"Backup not found: {backup_id}")
                return False
            
            if not record.backup_path.exists():
                self.logger.error(f"Backup directory not found: {record.backup_path}")
                return False
            
            try:
                # Restore configurations
                for config_name in record.configs:
                    config_file = record.backup_path / f"{config_name}.yaml"
                    
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        self.config_manager.set_config(config_name, config_data, user,
                                                     f"Restored from backup: {record.name}")
                
                self.logger.info(f"Restored backup: {record.name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to restore backup '{record.name}': {e}")
                return False

    def _delete_backup(self, backup_id: str):
        """Delete a backup"""
        record = self.backup_records.get(backup_id)
        if record:
            try:
                if record.backup_path.exists():
                    shutil.rmtree(record.backup_path)
                
                del self.backup_records[backup_id]
                self._save_backup_records()
                
                self.logger.info(f"Deleted backup: {record.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to delete backup '{record.name}': {e}")

    def get_backup_records(self) -> List[BackupRecord]:
        """Get all backup records"""
        return sorted(self.backup_records.values(), key=lambda x: x.created_at, reverse=True)

    def get_drift_detections(self, resolved: Optional[bool] = None) -> List[DriftDetection]:
        """Get drift detections"""
        detections = list(self.drift_detections.values())
        
        if resolved is not None:
            detections = [d for d in detections if d.corrected == resolved]
        
        return sorted(detections, key=lambda x: x.timestamp, reverse=True)

    def add_automation_rule(self, rule: AutomationRule) -> bool:
        """Add an automation rule"""
        with self._lock:
            if rule.id in self.automation_rules:
                return False
            
            self.automation_rules[rule.id] = rule
            self._save_automation_rules()
            
            self.logger.info(f"Added automation rule: {rule.name}")
            return True

    def update_automation_rule(self, rule_id: str, **kwargs) -> bool:
        """Update an automation rule"""
        with self._lock:
            rule = self.automation_rules.get(rule_id)
            if not rule:
                return False
            
            for field, value in kwargs.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            self._save_automation_rules()
            
            self.logger.info(f"Updated automation rule: {rule.name}")
            return True

    def delete_automation_rule(self, rule_id: str) -> bool:
        """Delete an automation rule"""
        with self._lock:
            if rule_id not in self.automation_rules:
                return False
            
            del self.automation_rules[rule_id]
            self._save_automation_rules()
            
            self.logger.info(f"Deleted automation rule: {rule_id}")
            return True

    def add_deployment_handler(self, handler: Callable[[DeploymentJob], None]):
        """Add a deployment handler"""
        self.deployment_handlers.append(handler)

    def remove_deployment_handler(self, handler: Callable[[DeploymentJob], None]):
        """Remove a deployment handler"""
        if handler in self.deployment_handlers:
            self.deployment_handlers.remove(handler)

    def add_drift_handler(self, handler: Callable[[DriftDetection], None]):
        """Add a drift handler"""
        self.drift_handlers.append(handler)

    def remove_drift_handler(self, handler: Callable[[DriftDetection], None]):
        """Remove a drift handler"""
        if handler in self.drift_handlers:
            self.drift_handlers.remove(handler)

    def get_automation_status(self) -> Dict[str, Any]:
        """Get automation system status"""
        pending_deployments = len([j for j in self.deployment_jobs.values() 
                                 if j.status == DeploymentStatus.PENDING])
        
        unresolved_drifts = len([d for d in self.drift_detections.values() 
                               if d.drift_detected and not d.corrected])
        
        return {
            'automation_active': self.automation_thread.is_alive() if self.automation_thread else False,
            'total_deployments': len(self.deployment_jobs),
            'pending_deployments': pending_deployments,
            'total_backups': len(self.backup_records),
            'unresolved_drifts': unresolved_drifts,
            'automation_rules': len(self.automation_rules),
            'enabled_rules': len([r for r in self.automation_rules.values() if r.enabled]),
            'deployment_handlers': len(self.deployment_handlers),
            'drift_handlers': len(self.drift_handlers)
        }

    def export_automation_data(self, export_path: Path):
        """Export automation data"""
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export deployment jobs
        self._save_deployment_jobs()
        shutil.copy2(self.deployments_path / "jobs.json", export_path / "deployment_jobs.json")
        
        # Export drift detections
        self._save_drift_detections()
        shutil.copy2(self.drift_path / "detections.json", export_path / "drift_detections.json")
        
        # Export backup records
        self._save_backup_records()
        shutil.copy2(self.backup_path / "records.json", export_path / "backup_records.json")
        
        # Export automation rules
        self._save_automation_rules()
        shutil.copy2(self.rules_path / "rules.json", export_path / "automation_rules.json")
        
        # Export status
        status = self.get_automation_status()
        with open(export_path / "automation_status.json", 'w') as f:
            json.dump(status, f, indent=2)
        
        self.logger.info(f"Automation data exported to {export_path}")

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_automation()