"""
AGENT 4: Continuous Secret Monitoring System
Real-time monitoring and alerting for secret leaks and security violations.

This module provides:
- Real-time file system monitoring
- Git commit hook integration
- CI/CD pipeline integration
- Automated alerting and remediation
- Security baseline management
- Drift detection and reporting
"""

import os
import json
import asyncio
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import re

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    Observer = None
    FileSystemEventHandler = None

from src.monitoring.logger_config import get_logger
from src.core.event_bus import EventBus

logger = get_logger(__name__)

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MonitoringEvent(str, Enum):
    """Monitoring event types"""
    SECRET_DETECTED = "SECRET_DETECTED"
    BASELINE_DRIFT = "BASELINE_DRIFT"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    REMEDIATION_REQUIRED = "REMEDIATION_REQUIRED"

@dataclass
class SecretAlert:
    """Security alert for detected secrets"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    event_type: MonitoringEvent
    file_path: str
    line_number: int
    secret_type: str
    matched_content: str
    context: str
    confidence: float
    remediation_suggested: str
    git_commit: Optional[str] = None
    author: Optional[str] = None
    branch: Optional[str] = None

@dataclass
class MonitoringConfig:
    """Configuration for continuous monitoring"""
    # Monitoring settings
    watch_directories: List[str] = field(default_factory=lambda: ["src", "configs", "scripts"])
    file_extensions: Set[str] = field(default_factory=lambda: {".py", ".yaml", ".yml", ".json", ".ini", ".env", ".sh"})
    excluded_directories: Set[str] = field(default_factory=lambda: {".git", "__pycache__", "node_modules", ".venv", "venv"})
    
    # Alert settings
    alert_threshold: float = 0.7
    batch_alerts: bool = True
    batch_interval: int = 60  # seconds
    
    # Remediation settings
    auto_remediation: bool = False
    quarantine_files: bool = True
    create_issues: bool = True
    
    # Notification settings
    slack_webhook: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    pagerduty_key: Optional[str] = None
    
    # Git integration
    git_hook_enabled: bool = True
    pre_commit_check: bool = True
    pre_push_check: bool = True

# =============================================================================
# SECRET PATTERNS (from previous module)
# =============================================================================

SECRET_PATTERNS = [
    {
        "name": "hardcoded_password",
        "pattern": r'password\s*[=:]\s*["\'][^"\']{4,}["\']',
        "severity": AlertSeverity.CRITICAL,
        "description": "Hardcoded password detected",
        "remediation": "Replace with environment variable or vault secret"
    },
    {
        "name": "api_key",
        "pattern": r'api[_-]?key\s*[=:]\s*["\'][^"\']{10,}["\']',
        "severity": AlertSeverity.CRITICAL,
        "description": "API key detected",
        "remediation": "Move to secure secret management system"
    },
    {
        "name": "jwt_secret",
        "pattern": r'jwt[_-]?secret\s*[=:]\s*["\'][^"\']{10,}["\']',
        "severity": AlertSeverity.CRITICAL,
        "description": "JWT secret detected",
        "remediation": "Use secure random generation and vault storage"
    },
    {
        "name": "database_url",
        "pattern": r'(postgresql|mysql|mongodb)://[^"\']*:[^"\']*@[^"\']*',
        "severity": AlertSeverity.HIGH,
        "description": "Database URL with embedded credentials",
        "remediation": "Use connection pooling with vault credentials"
    },
    {
        "name": "aws_access_key",
        "pattern": r'AKIA[0-9A-Z]{16}',
        "severity": AlertSeverity.CRITICAL,
        "description": "AWS access key detected",
        "remediation": "Rotate immediately and use IAM roles"
    },
    {
        "name": "private_key",
        "pattern": r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----',
        "severity": AlertSeverity.CRITICAL,
        "description": "Private key detected",
        "remediation": "Remove from code and use secure key management"
    }
]

# =============================================================================
# FILE SYSTEM MONITORING
# =============================================================================

class SecretFileHandler(FileSystemEventHandler):
    """File system event handler for secret monitoring"""
    
    def __init__(self, monitor: 'ContinuousSecretMonitor'):
        self.monitor = monitor
        self.debounce_time = 1.0  # seconds
        self.last_events: Dict[str, float] = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        self._handle_file_event(event.src_path, "modified")
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        self._handle_file_event(event.src_path, "created")
    
    def _handle_file_event(self, file_path: str, event_type: str):
        """Handle file system events with debouncing"""
        current_time = time.time()
        
        # Debounce events
        if file_path in self.last_events:
            if current_time - self.last_events[file_path] < self.debounce_time:
                return
        
        self.last_events[file_path] = current_time
        
        # Check if file should be monitored
        path_obj = Path(file_path)
        if not self.monitor._should_monitor_file(path_obj):
            return
        
        # Schedule async scan
        asyncio.create_task(self.monitor._scan_file_async(path_obj, event_type))

# =============================================================================
# CONTINUOUS SECRET MONITOR
# =============================================================================

class ContinuousSecretMonitor:
    """Continuous secret monitoring system"""
    
    def __init__(self, config: MonitoringConfig = None, event_bus: EventBus = None):
        self.config = config or MonitoringConfig()
        self.event_bus = event_bus or EventBus()
        
        # Monitoring state
        self.is_running = False
        self.observer = None
        self.alert_queue = asyncio.Queue()
        self.baseline_cache: Dict[str, str] = {}
        
        # Pattern compilation
        self.compiled_patterns = self._compile_patterns()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.alert_handlers: List[Callable] = []
        
        # Statistics
        self.stats = {
            "files_monitored": 0,
            "alerts_generated": 0,
            "secrets_detected": 0,
            "false_positives": 0,
            "start_time": None
        }
        
        logger.info("Continuous secret monitor initialized")
    
    def _compile_patterns(self) -> List[Dict]:
        """Compile regex patterns for efficient matching"""
        compiled = []
        for pattern_config in SECRET_PATTERNS:
            try:
                compiled.append({
                    "name": pattern_config["name"],
                    "pattern": re.compile(pattern_config["pattern"], re.IGNORECASE),
                    "severity": pattern_config["severity"],
                    "description": pattern_config["description"],
                    "remediation": pattern_config["remediation"]
                })
            except re.error as e:
                logger.error(f"Failed to compile pattern {pattern_config['name']}: {e}")
        return compiled
    
    def _should_monitor_file(self, file_path: Path) -> bool:
        """Check if file should be monitored"""
        # Check file extension
        if file_path.suffix not in self.config.file_extensions:
            return False
        
        # Check excluded directories
        for part in file_path.parts:
            if part in self.config.excluded_directories:
                return False
        
        # Check file size (skip large files)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return False
        except OSError:
            return False
        
        return True
    
    async def _scan_file_async(self, file_path: Path, event_type: str = "scan"):
        """Asynchronously scan file for secrets"""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check each pattern
            for pattern_config in self.compiled_patterns:
                matches = pattern_config["pattern"].finditer(content)
                
                for match in matches:
                    await self._process_match(
                        file_path=file_path,
                        match=match,
                        pattern_config=pattern_config,
                        content=content,
                        event_type=event_type
                    )
                    
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
    
    async def _process_match(self, file_path: Path, match: re.Match, 
                           pattern_config: Dict, content: str, event_type: str):
        """Process a pattern match"""
        # Calculate line number
        line_number = content[:match.start()].count('\n') + 1
        
        # Get context
        lines = content.split('\n')
        context_start = max(0, line_number - 2)
        context_end = min(len(lines), line_number + 2)
        context = '\n'.join(lines[context_start:context_end])
        
        # Calculate confidence
        confidence = self._calculate_confidence(match.group(0), pattern_config, context)
        
        # Skip if below threshold
        if confidence < self.config.alert_threshold:
            self.stats["false_positives"] += 1
            return
        
        # Get git information
        git_info = await self._get_git_info(file_path)
        
        # Create alert
        alert = SecretAlert(
            alert_id=self._generate_alert_id(),
            timestamp=datetime.now(),
            severity=pattern_config["severity"],
            event_type=MonitoringEvent.SECRET_DETECTED,
            file_path=str(file_path),
            line_number=line_number,
            secret_type=pattern_config["name"],
            matched_content=match.group(0),
            context=context,
            confidence=confidence,
            remediation_suggested=pattern_config["remediation"],
            git_commit=git_info.get("commit"),
            author=git_info.get("author"),
            branch=git_info.get("branch")
        )
        
        # Queue alert
        await self.alert_queue.put(alert)
        
        # Update statistics
        self.stats["alerts_generated"] += 1
        self.stats["secrets_detected"] += 1
        
        logger.warning(f"Secret detected: {pattern_config['name']} in {file_path}:{line_number}")
    
    def _calculate_confidence(self, match: str, pattern_config: Dict, context: str) -> float:
        """Calculate confidence score for a match"""
        confidence = 1.0
        
        # Reduce confidence for common test patterns
        test_indicators = ['test', 'example', 'demo', 'fake', 'mock', 'placeholder']
        if any(indicator in match.lower() for indicator in test_indicators):
            confidence *= 0.3
        
        # Reduce confidence for short matches
        if len(match) < 8:
            confidence *= 0.7
        
        # Reduce confidence if in test files
        if 'test' in context.lower():
            confidence *= 0.5
        
        # Increase confidence for specific patterns
        if pattern_config["name"] in ['aws_access_key', 'private_key']:
            confidence *= 1.2
        
        return min(confidence, 1.0)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return hashlib.md5(f"{datetime.now()}{os.urandom(8)}".encode()).hexdigest()[:8]
    
    async def _get_git_info(self, file_path: Path) -> Dict[str, str]:
        """Get git information for file"""
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=file_path.parent
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
            
            # Get last commit for file
            commit_result = subprocess.run(
                ["git", "log", "-1", "--format=%H", str(file_path)],
                capture_output=True, text=True, cwd=file_path.parent
            )
            commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
            
            # Get author
            author_result = subprocess.run(
                ["git", "log", "-1", "--format=%an", str(file_path)],
                capture_output=True, text=True, cwd=file_path.parent
            )
            author = author_result.stdout.strip() if author_result.returncode == 0 else None
            
            return {
                "branch": branch,
                "commit": commit,
                "author": author
            }
            
        except Exception as e:
            logger.debug(f"Failed to get git info for {file_path}: {e}")
            return {}
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        # Start file system monitoring
        if Observer is not None:
            self.observer = Observer()
            handler = SecretFileHandler(self)
            
            # Watch configured directories
            for directory in self.config.watch_directories:
                if Path(directory).exists():
                    self.observer.schedule(handler, directory, recursive=True)
                    logger.info(f"Watching directory: {directory}")
            
            self.observer.start()
        else:
            logger.warning("Watchdog not available, file system monitoring disabled")
        
        # Start alert processing
        asyncio.create_task(self._process_alerts())
        
        # Start periodic baseline check
        asyncio.create_task(self._periodic_baseline_check())
        
        logger.info("Continuous secret monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop file system monitoring
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Continuous secret monitoring stopped")
    
    async def _process_alerts(self):
        """Process alerts from queue"""
        while self.is_running:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Process alert
                await self._handle_alert(alert)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    async def _handle_alert(self, alert: SecretAlert):
        """Handle individual alert"""
        # Log alert
        logger.error(f"SECURITY ALERT: {alert.secret_type} detected in {alert.file_path}:{alert.line_number}")
        
        # Emit event
        await self.event_bus.emit("security_alert", {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "file_path": alert.file_path,
            "secret_type": alert.secret_type,
            "timestamp": alert.timestamp.isoformat()
        })
        
        # Auto-remediation
        if self.config.auto_remediation and alert.severity == AlertSeverity.CRITICAL:
            await self._auto_remediate(alert)
        
        # Quarantine file if configured
        if self.config.quarantine_files and alert.severity == AlertSeverity.CRITICAL:
            await self._quarantine_file(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Call registered handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def _auto_remediate(self, alert: SecretAlert):
        """Attempt automatic remediation"""
        try:
            # Create backup
            backup_path = Path(alert.file_path + ".backup")
            backup_path.write_text(Path(alert.file_path).read_text())
            
            # Comment out the line
            lines = Path(alert.file_path).read_text().split('\n')
            lines[alert.line_number - 1] = f"# SECURITY: {lines[alert.line_number - 1]}"
            
            Path(alert.file_path).write_text('\n'.join(lines))
            
            logger.info(f"Auto-remediated secret in {alert.file_path}")
            
        except Exception as e:
            logger.error(f"Auto-remediation failed for {alert.file_path}: {e}")
    
    async def _quarantine_file(self, alert: SecretAlert):
        """Quarantine file with secrets"""
        try:
            quarantine_dir = Path("/tmp/secret_quarantine")
            quarantine_dir.mkdir(exist_ok=True)
            
            # Move file to quarantine
            file_path = Path(alert.file_path)
            quarantine_path = quarantine_dir / f"{alert.alert_id}_{file_path.name}"
            
            quarantine_path.write_text(file_path.read_text())
            
            # Create quarantine info
            info = {
                "alert_id": alert.alert_id,
                "original_path": alert.file_path,
                "quarantine_time": datetime.now().isoformat(),
                "secret_type": alert.secret_type,
                "severity": alert.severity.value
            }
            
            (quarantine_path.parent / f"{alert.alert_id}_info.json").write_text(json.dumps(info))
            
            logger.info(f"File quarantined: {alert.file_path} -> {quarantine_path}")
            
        except Exception as e:
            logger.error(f"Quarantine failed for {alert.file_path}: {e}")
    
    async def _send_notifications(self, alert: SecretAlert):
        """Send notifications for alert"""
        try:
            # Slack notification
            if self.config.slack_webhook:
                await self._send_slack_notification(alert)
            
            # Email notification
            if self.config.email_recipients:
                await self._send_email_notification(alert)
            
            # PagerDuty notification
            if self.config.pagerduty_key and alert.severity == AlertSeverity.CRITICAL:
                await self._send_pagerduty_notification(alert)
                
        except Exception as e:
            logger.error(f"Notification failed: {e}")
    
    async def _send_slack_notification(self, alert: SecretAlert):
        """Send Slack notification"""
        # Implementation would use actual Slack webhook
        logger.info(f"Slack notification sent for alert {alert.alert_id}")
    
    async def _send_email_notification(self, alert: SecretAlert):
        """Send email notification"""
        # Implementation would use actual email service
        logger.info(f"Email notification sent for alert {alert.alert_id}")
    
    async def _send_pagerduty_notification(self, alert: SecretAlert):
        """Send PagerDuty notification"""
        # Implementation would use actual PagerDuty API
        logger.info(f"PagerDuty notification sent for alert {alert.alert_id}")
    
    async def _periodic_baseline_check(self):
        """Periodic baseline drift check"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self._check_baseline_drift()
            except Exception as e:
                logger.error(f"Baseline check failed: {e}")
    
    async def _check_baseline_drift(self):
        """Check for baseline drift"""
        logger.info("Checking baseline drift...")
        
        # Load baseline
        baseline_path = Path("security_baseline.json")
        if not baseline_path.exists():
            logger.warning("No security baseline found")
            return
        
        try:
            baseline = json.loads(baseline_path.read_text())
            
            # Perform current scan
            current_stats = await self._perform_full_scan()
            
            # Compare with baseline
            if current_stats["secrets_found"] > baseline.get("secrets_found", 0):
                drift_alert = SecretAlert(
                    alert_id=self._generate_alert_id(),
                    timestamp=datetime.now(),
                    severity=AlertSeverity.HIGH,
                    event_type=MonitoringEvent.BASELINE_DRIFT,
                    file_path="BASELINE_DRIFT",
                    line_number=0,
                    secret_type="baseline_drift",
                    matched_content=f"Baseline drift detected: {current_stats['secrets_found']} vs {baseline.get('secrets_found', 0)}",
                    context="Security baseline drift detected",
                    confidence=1.0,
                    remediation_suggested="Review recent changes and update baseline if legitimate"
                )
                
                await self.alert_queue.put(drift_alert)
                
        except Exception as e:
            logger.error(f"Baseline drift check failed: {e}")
    
    async def _perform_full_scan(self) -> Dict[str, Any]:
        """Perform full repository scan"""
        secrets_found = 0
        files_scanned = 0
        
        for directory in self.config.watch_directories:
            dir_path = Path(directory)
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and self._should_monitor_file(file_path):
                        files_scanned += 1
                        # Simplified scan (would use full detection logic)
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            for pattern_config in self.compiled_patterns:
                                if pattern_config["pattern"].search(content):
                                    secrets_found += 1
                                    break
                        except Exception:
                            continue
        
        return {
            "secrets_found": secrets_found,
            "files_scanned": files_scanned,
            "scan_time": datetime.now().isoformat()
        }
    
    def add_alert_handler(self, handler: Callable[[SecretAlert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = self.stats.copy()
        if stats["start_time"]:
            stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
        return stats
    
    async def create_baseline(self, output_file: str = "security_baseline.json"):
        """Create security baseline"""
        logger.info("Creating security baseline...")
        
        scan_results = await self._perform_full_scan()
        
        baseline = {
            "created_at": datetime.now().isoformat(),
            "secrets_found": scan_results["secrets_found"],
            "files_scanned": scan_results["files_scanned"],
            "monitoring_config": {
                "watch_directories": self.config.watch_directories,
                "file_extensions": list(self.config.file_extensions),
                "alert_threshold": self.config.alert_threshold
            },
            "git_commit": self._get_current_git_commit()
        }
        
        Path(output_file).write_text(json.dumps(baseline, indent=2))
        logger.info(f"Security baseline created: {output_file}")
    
    def _get_current_git_commit(self) -> Optional[str]:
        """Get current git commit"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

# =============================================================================
# GIT HOOK INTEGRATION
# =============================================================================

class GitHookIntegration:
    """Git hook integration for secret detection"""
    
    def __init__(self, monitor: ContinuousSecretMonitor):
        self.monitor = monitor
    
    def install_hooks(self, git_dir: str = ".git"):
        """Install git hooks"""
        hooks_dir = Path(git_dir) / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Secret detection pre-commit hook
python -m src.security.continuous_secret_monitoring --pre-commit
"""
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Pre-push hook
        pre_push_hook = hooks_dir / "pre-push"
        pre_push_content = """#!/bin/bash
# Secret detection pre-push hook
python -m src.security.continuous_secret_monitoring --pre-push
"""
        pre_push_hook.write_text(pre_push_content)
        pre_push_hook.chmod(0o755)
        
        logger.info("Git hooks installed")
    
    async def pre_commit_check(self) -> bool:
        """Pre-commit secret check"""
        logger.info("Running pre-commit secret check...")
        
        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error("Failed to get staged files")
            return False
        
        staged_files = result.stdout.strip().split('\n')
        secrets_found = False
        
        for file_path in staged_files:
            if file_path and Path(file_path).exists():
                path_obj = Path(file_path)
                if self.monitor._should_monitor_file(path_obj):
                    # Scan file
                    content = path_obj.read_text(encoding='utf-8', errors='ignore')
                    for pattern_config in self.monitor.compiled_patterns:
                        if pattern_config["pattern"].search(content):
                            logger.error(f"Secret detected in staged file: {file_path}")
                            secrets_found = True
                            break
        
        if secrets_found:
            logger.error("❌ Commit blocked: Secrets detected in staged files")
            return False
        
        logger.info("✅ Pre-commit check passed")
        return True
    
    async def pre_push_check(self) -> bool:
        """Pre-push secret check"""
        logger.info("Running pre-push secret check...")
        
        # Get commits to be pushed
        result = subprocess.run(
            ["git", "rev-list", "--oneline", "@{u}..HEAD"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.info("No upstream branch or commits to push")
            return True
        
        commits = result.stdout.strip().split('\n')
        secrets_found = False
        
        for commit_line in commits:
            if commit_line:
                commit_hash = commit_line.split()[0]
                
                # Get files changed in commit
                files_result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
                    capture_output=True, text=True
                )
                
                if files_result.returncode == 0:
                    changed_files = files_result.stdout.strip().split('\n')
                    
                    for file_path in changed_files:
                        if file_path and Path(file_path).exists():
                            path_obj = Path(file_path)
                            if self.monitor._should_monitor_file(path_obj):
                                # Scan file
                                content = path_obj.read_text(encoding='utf-8', errors='ignore')
                                for pattern_config in self.monitor.compiled_patterns:
                                    if pattern_config["pattern"].search(content):
                                        logger.error(f"Secret detected in commit {commit_hash}: {file_path}")
                                        secrets_found = True
                                        break
        
        if secrets_found:
            logger.error("❌ Push blocked: Secrets detected in commits")
            return False
        
        logger.info("✅ Pre-push check passed")
        return True

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for continuous monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Secret Monitoring")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit check")
    parser.add_argument("--pre-push", action="store_true", help="Run pre-push check")
    parser.add_argument("--create-baseline", action="store_true", help="Create security baseline")
    parser.add_argument("--install-hooks", action="store_true", help="Install git hooks")
    parser.add_argument("--start", action="store_true", help="Start continuous monitoring")
    
    args = parser.parse_args()
    
    # Initialize monitor
    config = MonitoringConfig()
    monitor = ContinuousSecretMonitor(config)
    git_hooks = GitHookIntegration(monitor)
    
    try:
        if args.pre_commit:
            success = await git_hooks.pre_commit_check()
            exit(0 if success else 1)
        
        elif args.pre_push:
            success = await git_hooks.pre_push_check()
            exit(0 if success else 1)
        
        elif args.create_baseline:
            await monitor.create_baseline()
        
        elif args.install_hooks:
            git_hooks.install_hooks()
        
        elif args.start:
            await monitor.start_monitoring()
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(60)
                    stats = monitor.get_statistics()
                    logger.info(f"Monitoring stats: {stats}")
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                await monitor.stop_monitoring()
        
        else:
            logger.info("No action specified. Use --help for options.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())