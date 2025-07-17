"""
Continuous Benchmarking Pipeline

This module provides a comprehensive continuous benchmarking pipeline that automatically
runs performance tests, tracks baselines, detects regressions, and generates reports.

Features:
- Automated benchmark scheduling
- Baseline tracking and comparison
- Performance regression detection
- CI/CD integration
- Automated reporting
- Alert system for performance issues
- Multi-environment support
- Historical trend analysis

Author: Performance Validation Agent
"""

import asyncio
import schedule
import time
import json
import sqlite3
import subprocess
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import structlog
from enum import Enum
import git
import yaml
import requests
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class BenchmarkStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class BenchmarkConfiguration:
    """Benchmark configuration"""
    name: str
    test_module: str
    test_function: str
    schedule_cron: str
    enabled: bool = True
    timeout_seconds: int = 300
    baseline_branches: List[str] = field(default_factory=lambda: ["main", "master"])
    environment: str = "development"
    parameters: Dict[str, Any] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class BenchmarkExecution:
    """Benchmark execution record"""
    id: str
    configuration: BenchmarkConfiguration
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    environment_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionAlert:
    """Performance regression alert"""
    benchmark_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str
    timestamp: datetime
    git_commit: Optional[str] = None
    recommendation: str = ""

class ContinuousBenchmarkingPipeline:
    """
    Continuous benchmarking pipeline for automated performance testing
    and regression detection across the entire system.
    """

    def __init__(self, config_path: str = "configs/performance_regression_config.yaml"):
        self.config_path = Path(config_path)
        self.db_path = "ci_performance.db"
        self.benchmark_configurations = {}
        self.active_executions = {}
        self.execution_history = []
        self.alert_history = []
        
        # Pipeline status
        self.pipeline_active = False
        self.scheduler_thread = None
        
        # Load configuration
        self._load_configuration()
        
        # Initialize database
        self._init_database()
        
        # Initialize Git repository
        try:
            self.git_repo = git.Repo(".")
        except git.InvalidGitRepositoryError:
            self.git_repo = None
            logger.warning("Not in a Git repository, Git integration disabled")
        
        # Initialize notification system
        self._init_notification_system()
        
        logger.info("Continuous benchmarking pipeline initialized",
                   configurations=len(self.benchmark_configurations))

    def _load_configuration(self):
        """Load benchmark configurations from YAML file"""
        if not self.config_path.exists():
            # Create default configuration
            default_config = {
                'benchmarks': {
                    'strategic_inference': {
                        'test_module': 'tests.performance.test_strategic_inference',
                        'test_function': 'test_strategic_inference_latency',
                        'schedule_cron': '0 */4 * * *',  # Every 4 hours
                        'enabled': True,
                        'timeout_seconds': 300,
                        'baseline_branches': ['main', 'master'],
                        'environment': 'ci',
                        'parameters': {'iterations': 1000},
                        'notification_channels': ['email', 'slack']
                    },
                    'tactical_inference': {
                        'test_module': 'tests.performance.test_tactical_inference',
                        'test_function': 'test_tactical_inference_latency',
                        'schedule_cron': '0 */4 * * *',
                        'enabled': True,
                        'timeout_seconds': 300,
                        'baseline_branches': ['main', 'master'],
                        'environment': 'ci',
                        'parameters': {'iterations': 1000},
                        'notification_channels': ['email', 'slack']
                    },
                    'end_to_end_pipeline': {
                        'test_module': 'tests.performance.test_end_to_end',
                        'test_function': 'test_complete_pipeline_performance',
                        'schedule_cron': '0 */6 * * *',  # Every 6 hours
                        'enabled': True,
                        'timeout_seconds': 600,
                        'baseline_branches': ['main', 'master'],
                        'environment': 'ci',
                        'parameters': {'duration_seconds': 60},
                        'notification_channels': ['email', 'slack']
                    },
                    'database_recovery': {
                        'test_module': 'tests.performance.test_database_recovery',
                        'test_function': 'test_database_rto',
                        'schedule_cron': '0 */12 * * *',  # Every 12 hours
                        'enabled': True,
                        'timeout_seconds': 300,
                        'baseline_branches': ['main', 'master'],
                        'environment': 'ci',
                        'parameters': {},
                        'notification_channels': ['email', 'slack']
                    },
                    'trading_engine_recovery': {
                        'test_module': 'tests.performance.test_trading_engine_recovery',
                        'test_function': 'test_trading_engine_rto',
                        'schedule_cron': '0 */12 * * *',  # Every 12 hours
                        'enabled': True,
                        'timeout_seconds': 300,
                        'baseline_branches': ['main', 'master'],
                        'environment': 'ci',
                        'parameters': {},
                        'notification_channels': ['email', 'slack']
                    }
                },
                'notification': {
                    'email': {
                        'enabled': True,
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'sender_email': 'performance@grandmodel.com',
                        'sender_password': 'your_app_password',
                        'recipients': ['team@grandmodel.com']
                    },
                    'slack': {
                        'enabled': True,
                        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
                    }
                },
                'regression_detection': {
                    'significance_threshold': 0.05,
                    'minimum_samples': 5,
                    'regression_threshold_percent': 20.0,
                    'baseline_update_frequency': 'daily'
                }
            }
            
            # Create config directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            logger.info("Created default configuration file", path=str(self.config_path))
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse benchmark configurations
        for name, bench_config in config.get('benchmarks', {}).items():
            self.benchmark_configurations[name] = BenchmarkConfiguration(
                name=name,
                test_module=bench_config['test_module'],
                test_function=bench_config['test_function'],
                schedule_cron=bench_config['schedule_cron'],
                enabled=bench_config.get('enabled', True),
                timeout_seconds=bench_config.get('timeout_seconds', 300),
                baseline_branches=bench_config.get('baseline_branches', ['main']),
                environment=bench_config.get('environment', 'ci'),
                parameters=bench_config.get('parameters', {}),
                notification_channels=bench_config.get('notification_channels', [])
            )
        
        # Store other configuration
        self.notification_config = config.get('notification', {})
        self.regression_config = config.get('regression_detection', {})

    def _init_database(self):
        """Initialize CI performance database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Benchmark executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_executions (
                id TEXT PRIMARY KEY,
                configuration_name TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL,
                results TEXT,
                error_message TEXT,
                git_commit TEXT,
                git_branch TEXT,
                environment_info TEXT
            )
        """)
        
        # Performance baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                benchmark_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                baseline_std REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                PRIMARY KEY (benchmark_name, metric_name)
            )
        """)
        
        # Regression alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                regression_percent REAL NOT NULL,
                severity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                recommendation TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Pipeline status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                active_executions INTEGER NOT NULL,
                total_executions INTEGER NOT NULL,
                recent_alerts INTEGER NOT NULL,
                system_health_score REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()

    def _init_notification_system(self):
        """Initialize notification system"""
        self.notification_handlers = {}
        
        # Email notification handler
        if self.notification_config.get('email', {}).get('enabled', False):
            self.notification_handlers['email'] = self._send_email_notification
        
        # Slack notification handler
        if self.notification_config.get('slack', {}).get('enabled', False):
            self.notification_handlers['slack'] = self._send_slack_notification

    def start_pipeline(self):
        """Start the continuous benchmarking pipeline"""
        if self.pipeline_active:
            logger.warning("Pipeline is already active")
            return
        
        self.pipeline_active = True
        
        # Schedule benchmarks
        for config in self.benchmark_configurations.values():
            if config.enabled:
                schedule.every().day.at("00:00").do(
                    self._schedule_benchmark_execution, config.name
                )
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Continuous benchmarking pipeline started",
                   active_benchmarks=len([c for c in self.benchmark_configurations.values() if c.enabled]))

    def stop_pipeline(self):
        """Stop the continuous benchmarking pipeline"""
        if not self.pipeline_active:
            return
        
        self.pipeline_active = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Continuous benchmarking pipeline stopped")

    def _run_scheduler(self):
        """Run the benchmark scheduler"""
        while self.pipeline_active:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _schedule_benchmark_execution(self, benchmark_name: str):
        """Schedule a benchmark execution"""
        if benchmark_name not in self.benchmark_configurations:
            logger.error("Unknown benchmark configuration", benchmark_name=benchmark_name)
            return
        
        config = self.benchmark_configurations[benchmark_name]
        
        # Create execution record
        execution_id = f"{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = BenchmarkExecution(
            id=execution_id,
            configuration=config,
            status=BenchmarkStatus.PENDING,
            start_time=datetime.now(),
            git_commit=self._get_current_git_commit(),
            git_branch=self._get_current_git_branch(),
            environment_info=self._get_environment_info()
        )
        
        # Execute benchmark in background
        threading.Thread(
            target=self._execute_benchmark,
            args=(execution,),
            daemon=True
        ).start()

    def _execute_benchmark(self, execution: BenchmarkExecution):
        """Execute a benchmark"""
        logger.info("Starting benchmark execution",
                   benchmark_name=execution.configuration.name,
                   execution_id=execution.id)
        
        execution.status = BenchmarkStatus.RUNNING
        self.active_executions[execution.id] = execution
        
        try:
            # Run the benchmark
            start_time = time.time()
            
            # Import and run the test function
            module_path = execution.configuration.test_module
            function_name = execution.configuration.test_function
            
            # Dynamic import
            module = __import__(module_path, fromlist=[function_name])
            test_function = getattr(module, function_name)
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(test_function):
                # Async function
                results = asyncio.run(
                    asyncio.wait_for(
                        test_function(**execution.configuration.parameters),
                        timeout=execution.configuration.timeout_seconds
                    )
                )
            else:
                # Sync function
                results = test_function(**execution.configuration.parameters)
            
            end_time = time.time()
            execution.duration_seconds = end_time - start_time
            execution.end_time = datetime.now()
            execution.results = results
            execution.status = BenchmarkStatus.COMPLETED
            
            # Store execution record
            self._store_execution_record(execution)
            
            # Update baselines if from baseline branch
            if execution.git_branch in execution.configuration.baseline_branches:
                self._update_baselines(execution)
            
            # Check for regressions
            self._check_for_regressions(execution)
            
            logger.info("Benchmark execution completed",
                       benchmark_name=execution.configuration.name,
                       execution_id=execution.id,
                       duration_seconds=execution.duration_seconds)
            
        except Exception as e:
            execution.status = BenchmarkStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            execution.duration_seconds = time.time() - start_time if 'start_time' in locals() else 0
            
            self._store_execution_record(execution)
            
            logger.error("Benchmark execution failed",
                        benchmark_name=execution.configuration.name,
                        execution_id=execution.id,
                        error=str(e))
            
            # Send failure notification
            self._send_failure_notification(execution)
        
        finally:
            # Clean up
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
            
            self.execution_history.append(execution)
            
            # Keep only recent executions in memory
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]

    def _store_execution_record(self, execution: BenchmarkExecution):
        """Store execution record in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO benchmark_executions 
            (id, configuration_name, status, start_time, end_time, duration_seconds,
             results, error_message, git_commit, git_branch, environment_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.id,
            execution.configuration.name,
            execution.status.value,
            execution.start_time.isoformat(),
            execution.end_time.isoformat() if execution.end_time else None,
            execution.duration_seconds,
            json.dumps(execution.results),
            execution.error_message,
            execution.git_commit,
            execution.git_branch,
            json.dumps(execution.environment_info)
        ))
        
        conn.commit()
        conn.close()

    def _update_baselines(self, execution: BenchmarkExecution):
        """Update performance baselines from execution results"""
        if execution.status != BenchmarkStatus.COMPLETED:
            return
        
        benchmark_name = execution.configuration.name
        results = execution.results
        
        # Extract performance metrics from results
        metrics = {}
        if isinstance(results, dict):
            # Common metric names
            metric_mappings = {
                'avg_latency_ms': 'avg_latency_ms',
                'p95_latency_ms': 'p95_latency_ms',
                'p99_latency_ms': 'p99_latency_ms',
                'throughput_ops_per_sec': 'throughput_ops_per_sec',
                'memory_usage_mb': 'memory_usage_mb',
                'cpu_percent': 'cpu_percent'
            }
            
            for result_key, metric_name in metric_mappings.items():
                if result_key in results:
                    metrics[metric_name] = float(results[result_key])
        
        # Update baselines
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, value in metrics.items():
            # Get recent values for this metric
            cursor.execute("""
                SELECT results FROM benchmark_executions 
                WHERE configuration_name = ? AND git_branch IN ({})
                ORDER BY start_time DESC 
                LIMIT 20
            """.format(','.join(['?'] * len(execution.configuration.baseline_branches))),
                      [benchmark_name] + execution.configuration.baseline_branches)
            
            recent_results = cursor.fetchall()
            recent_values = []
            
            for result_json, in recent_results:
                try:
                    result_data = json.loads(result_json)
                    if metric_name in result_data:
                        recent_values.append(float(result_data[metric_name]))
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
            
            # Update baseline if we have enough samples
            if len(recent_values) >= self.regression_config.get('minimum_samples', 5):
                baseline_value = np.mean(recent_values)
                baseline_std = np.std(recent_values)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_baselines 
                    (benchmark_name, metric_name, baseline_value, baseline_std, sample_count,
                     last_updated, git_commit, git_branch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    benchmark_name,
                    metric_name,
                    baseline_value,
                    baseline_std,
                    len(recent_values),
                    datetime.now().isoformat(),
                    execution.git_commit,
                    execution.git_branch
                ))
                
                logger.info("Updated baseline",
                           benchmark_name=benchmark_name,
                           metric_name=metric_name,
                           baseline_value=baseline_value,
                           sample_count=len(recent_values))
        
        conn.commit()
        conn.close()

    def _check_for_regressions(self, execution: BenchmarkExecution):
        """Check for performance regressions"""
        if execution.status != BenchmarkStatus.COMPLETED:
            return
        
        benchmark_name = execution.configuration.name
        results = execution.results
        
        if not isinstance(results, dict):
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current baselines
        cursor.execute("""
            SELECT metric_name, baseline_value, baseline_std 
            FROM performance_baselines 
            WHERE benchmark_name = ?
        """, (benchmark_name,))
        
        baselines = {row[0]: {'value': row[1], 'std': row[2]} for row in cursor.fetchall()}
        
        # Check metrics against baselines
        for metric_name, baseline_info in baselines.items():
            if metric_name in results:
                current_value = float(results[metric_name])
                baseline_value = baseline_info['value']
                baseline_std = baseline_info['std']
                
                # Calculate regression
                if baseline_value > 0:
                    regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    regression_percent = 0
                
                # Check if regression is significant
                threshold = self.regression_config.get('regression_threshold_percent', 20.0)
                
                if regression_percent > threshold:
                    # Determine severity
                    if regression_percent > 100:
                        severity = "CRITICAL"
                    elif regression_percent > 50:
                        severity = "HIGH"
                    elif regression_percent > 25:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"
                    
                    # Create regression alert
                    alert = RegressionAlert(
                        benchmark_name=benchmark_name,
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        regression_percent=regression_percent,
                        severity=severity,
                        timestamp=datetime.now(),
                        git_commit=execution.git_commit,
                        recommendation=self._generate_regression_recommendation(
                            benchmark_name, metric_name, regression_percent
                        )
                    )
                    
                    # Store alert
                    self._store_regression_alert(alert)
                    
                    # Send notification
                    self._send_regression_notification(alert)
        
        conn.close()

    def _store_regression_alert(self, alert: RegressionAlert):
        """Store regression alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO regression_alerts 
            (benchmark_name, metric_name, current_value, baseline_value, regression_percent,
             severity, timestamp, git_commit, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.benchmark_name,
            alert.metric_name,
            alert.current_value,
            alert.baseline_value,
            alert.regression_percent,
            alert.severity,
            alert.timestamp.isoformat(),
            alert.git_commit,
            alert.recommendation
        ))
        
        conn.commit()
        conn.close()
        
        self.alert_history.append(alert)
        
        logger.warning("Performance regression detected",
                      benchmark_name=alert.benchmark_name,
                      metric_name=alert.metric_name,
                      severity=alert.severity,
                      regression_percent=alert.regression_percent)

    def _generate_regression_recommendation(self, benchmark_name: str, metric_name: str, 
                                          regression_percent: float) -> str:
        """Generate regression recommendation"""
        recommendations = []
        
        if regression_percent > 100:
            recommendations.append("CRITICAL: Immediate investigation required")
        elif regression_percent > 50:
            recommendations.append("HIGH: Review recent changes and optimize")
        else:
            recommendations.append("MEDIUM: Monitor performance trends")
        
        if "latency" in metric_name:
            recommendations.append("Consider algorithmic optimization")
        elif "memory" in metric_name:
            recommendations.append("Investigate memory leaks")
        elif "throughput" in metric_name:
            recommendations.append("Review resource utilization")
        
        return "; ".join(recommendations)

    def _send_regression_notification(self, alert: RegressionAlert):
        """Send regression notification"""
        message = f"""
Performance Regression Alert

Benchmark: {alert.benchmark_name}
Metric: {alert.metric_name}
Severity: {alert.severity}
Current Value: {alert.current_value:.2f}
Baseline Value: {alert.baseline_value:.2f}
Regression: {alert.regression_percent:.1f}%
Git Commit: {alert.git_commit or 'Unknown'}
Recommendation: {alert.recommendation}

Timestamp: {alert.timestamp.isoformat()}
"""
        
        # Send to configured channels
        for config in self.benchmark_configurations.values():
            if alert.benchmark_name == config.name:
                for channel in config.notification_channels:
                    if channel in self.notification_handlers:
                        self.notification_handlers[channel](
                            f"Performance Regression: {alert.benchmark_name}",
                            message
                        )

    def _send_failure_notification(self, execution: BenchmarkExecution):
        """Send benchmark failure notification"""
        message = f"""
Benchmark Execution Failed

Benchmark: {execution.configuration.name}
Execution ID: {execution.id}
Error: {execution.error_message}
Duration: {execution.duration_seconds:.2f}s
Git Commit: {execution.git_commit or 'Unknown'}
Git Branch: {execution.git_branch or 'Unknown'}

Timestamp: {execution.start_time.isoformat()}
"""
        
        # Send to configured channels
        for channel in execution.configuration.notification_channels:
            if channel in self.notification_handlers:
                self.notification_handlers[channel](
                    f"Benchmark Failure: {execution.configuration.name}",
                    message
                )

    def _send_email_notification(self, subject: str, message: str):
        """Send email notification"""
        try:
            email_config = self.notification_config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent", subject=subject)
            
        except Exception as e:
            logger.error("Failed to send email notification", error=str(e))

    def _send_slack_notification(self, subject: str, message: str):
        """Send Slack notification"""
        try:
            slack_config = self.notification_config['slack']
            
            payload = {
                'text': f"*{subject}*\n```{message}```"
            }
            
            response = requests.post(slack_config['webhook_url'], json=payload)
            response.raise_for_status()
            
            logger.info("Slack notification sent", subject=subject)
            
        except Exception as e:
            logger.error("Failed to send Slack notification", error=str(e))

    def _get_current_git_commit(self) -> Optional[str]:
        """Get current Git commit hash"""
        if self.git_repo:
            try:
                return self.git_repo.head.commit.hexsha
            except Exception:
                pass
        return None

    def _get_current_git_branch(self) -> Optional[str]:
        """Get current Git branch"""
        if self.git_repo:
            try:
                return self.git_repo.active_branch.name
            except Exception:
                pass
        return None

    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'python_version': subprocess.check_output(['python', '--version']).decode().strip(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'platform': subprocess.check_output(['uname', '-a']).decode().strip(),
            'timestamp': datetime.now().isoformat()
        }

    def trigger_benchmark(self, benchmark_name: str) -> str:
        """Manually trigger a benchmark execution"""
        if benchmark_name not in self.benchmark_configurations:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        execution_id = f"{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_manual"
        config = self.benchmark_configurations[benchmark_name]
        
        execution = BenchmarkExecution(
            id=execution_id,
            configuration=config,
            status=BenchmarkStatus.PENDING,
            start_time=datetime.now(),
            git_commit=self._get_current_git_commit(),
            git_branch=self._get_current_git_branch(),
            environment_info=self._get_environment_info()
        )
        
        # Execute in background
        threading.Thread(
            target=self._execute_benchmark,
            args=(execution,),
            daemon=True
        ).start()
        
        logger.info("Manual benchmark triggered",
                   benchmark_name=benchmark_name,
                   execution_id=execution_id)
        
        return execution_id

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent executions
        cursor.execute("""
            SELECT status, COUNT(*) FROM benchmark_executions 
            WHERE start_time >= datetime('now', '-24 hours')
            GROUP BY status
        """)
        
        status_counts = dict(cursor.fetchall())
        
        # Get recent alerts
        cursor.execute("""
            SELECT COUNT(*) FROM regression_alerts 
            WHERE timestamp >= datetime('now', '-24 hours')
        """)
        
        recent_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'pipeline_active': self.pipeline_active,
            'active_executions': len(self.active_executions),
            'configured_benchmarks': len(self.benchmark_configurations),
            'enabled_benchmarks': len([c for c in self.benchmark_configurations.values() if c.enabled]),
            'recent_executions': status_counts,
            'recent_alerts': recent_alerts,
            'last_update': datetime.now().isoformat()
        }

    def generate_pipeline_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get execution statistics
        cursor.execute("""
            SELECT configuration_name, status, COUNT(*), AVG(duration_seconds)
            FROM benchmark_executions 
            WHERE start_time >= ?
            GROUP BY configuration_name, status
        """, (cutoff_time.isoformat(),))
        
        execution_stats = cursor.fetchall()
        
        # Get regression alerts
        cursor.execute("""
            SELECT benchmark_name, metric_name, severity, COUNT(*)
            FROM regression_alerts 
            WHERE timestamp >= ?
            GROUP BY benchmark_name, metric_name, severity
        """, (cutoff_time.isoformat(),))
        
        alert_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_period_hours': hours,
            'pipeline_status': self.get_pipeline_status(),
            'execution_statistics': [
                {
                    'benchmark_name': row[0],
                    'status': row[1],
                    'count': row[2],
                    'avg_duration_seconds': row[3]
                }
                for row in execution_stats
            ],
            'alert_statistics': [
                {
                    'benchmark_name': row[0],
                    'metric_name': row[1],
                    'severity': row[2],
                    'count': row[3]
                }
                for row in alert_stats
            ],
            'configuration_summary': {
                name: {
                    'enabled': config.enabled,
                    'schedule': config.schedule_cron,
                    'timeout_seconds': config.timeout_seconds,
                    'environment': config.environment
                }
                for name, config in self.benchmark_configurations.items()
            }
        }

    def cleanup(self):
        """Cleanup resources"""
        self.stop_pipeline()
        logger.info("Continuous benchmarking pipeline cleanup completed")

    def __del__(self):
        """Destructor"""
        self.cleanup()


# Global instance
continuous_benchmarking = ContinuousBenchmarkingPipeline()