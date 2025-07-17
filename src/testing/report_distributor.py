"""
Report Distribution and Notification System
Automated distribution of test reports via email, Slack, and other channels
"""

import smtplib
import ssl
import json
import asyncio
import aiohttp
import aiofiles
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import jinja2
import base64
import hashlib
import sqlite3
from .advanced_test_reporting import TestSuite
from .coverage_analyzer import CoverageReport


class NotificationChannel(Enum):
    """Supported notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    FILE = "file"
    DATABASE = "database"


class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NotificationTemplate:
    """Template for notifications"""
    name: str
    subject: str
    body: str
    format: str  # 'html' or 'text'
    channels: List[str]
    conditions: Dict[str, Any]


@dataclass
class NotificationRecipient:
    """Notification recipient configuration"""
    name: str
    email: Optional[str] = None
    slack_user_id: Optional[str] = None
    teams_user_id: Optional[str] = None
    notification_levels: List[str] = None
    report_formats: List[str] = None
    schedule: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.notification_levels is None:
            self.notification_levels = ["error", "critical"]
        if self.report_formats is None:
            self.report_formats = ["html", "json"]
        if self.schedule is None:
            self.schedule = {"always": True}


@dataclass
class DistributionConfig:
    """Configuration for report distribution"""
    enabled: bool = True
    recipients: List[NotificationRecipient] = None
    templates: List[NotificationTemplate] = None
    email_config: Dict[str, Any] = None
    slack_config: Dict[str, Any] = None
    teams_config: Dict[str, Any] = None
    webhook_config: Dict[str, Any] = None
    archive_config: Dict[str, Any] = None
    retry_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []
        if self.templates is None:
            self.templates = []
        if self.email_config is None:
            self.email_config = {}
        if self.slack_config is None:
            self.slack_config = {}
        if self.teams_config is None:
            self.teams_config = {}
        if self.webhook_config is None:
            self.webhook_config = {}
        if self.archive_config is None:
            self.archive_config = {"enabled": True, "retention_days": 30}
        if self.retry_config is None:
            self.retry_config = {"max_retries": 3, "delay_seconds": 60}


class ReportDistributor:
    """Handles distribution of test reports to various channels"""
    
    def __init__(self, config: DistributionConfig, output_dir: str = "distribution_logs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_path = self.output_dir / "distribution_history.db"
        self._init_database()
        
        # Setup Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=True
        )
        
        # Create default templates
        self._create_default_templates()
    
    def _init_database(self):
        """Initialize database for distribution tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS distribution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                distribution_id TEXT UNIQUE,
                suite_name TEXT,
                generated_at TIMESTAMP,
                channels TEXT,
                recipients TEXT,
                status TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                reports_sent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                distribution_id TEXT,
                channel TEXT,
                recipient TEXT,
                status TEXT,
                message TEXT,
                response_data TEXT,
                sent_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (distribution_id) REFERENCES distribution_history (distribution_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS template_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_name TEXT,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_default_templates(self):
        """Create default notification templates"""
        
        # Success template
        success_template = NotificationTemplate(
            name="test_success",
            subject="✅ Test Suite Passed: {{ suite.suite_name }}",
            body="""
            <h2>Test Suite Successfully Completed</h2>
            <p><strong>Suite:</strong> {{ suite.suite_name }}</p>
            <p><strong>Duration:</strong> {{ suite.total_duration|round(2) }} seconds</p>
            <p><strong>Results:</strong> {{ suite.passed }} passed, {{ suite.failed }} failed, {{ suite.skipped }} skipped</p>
            <p><strong>Success Rate:</strong> {{ suite.success_rate|round(1) }}%</p>
            <p><strong>Coverage:</strong> {{ suite.coverage_percentage|round(1) }}%</p>
            
            {% if recommendations %}
            <h3>Recommendations:</h3>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            <p><em>Generated at {{ suite.end_time.strftime('%Y-%m-%d %H:%M:%S') }}</em></p>
            """,
            format="html",
            channels=["email", "slack"],
            conditions={"success_rate": {"min": 95}}
        )
        
        # Warning template
        warning_template = NotificationTemplate(
            name="test_warning",
            subject="⚠️ Test Suite Warning: {{ suite.suite_name }}",
            body="""
            <h2>Test Suite Completed with Issues</h2>
            <p><strong>Suite:</strong> {{ suite.suite_name }}</p>
            <p><strong>Duration:</strong> {{ suite.total_duration|round(2) }} seconds</p>
            <p><strong>Results:</strong> {{ suite.passed }} passed, {{ suite.failed }} failed, {{ suite.skipped }} skipped</p>
            <p><strong>Success Rate:</strong> {{ suite.success_rate|round(1) }}%</p>
            <p><strong>Coverage:</strong> {{ suite.coverage_percentage|round(1) }}%</p>
            
            {% if suite.failed > 0 %}
            <h3>Failed Tests:</h3>
            <ul>
                {% for result in suite.results %}
                {% if result.status.value == 'failed' %}
                <li><strong>{{ result.test_name }}</strong> ({{ result.test_module }})</li>
                {% endif %}
                {% endfor %}
            </ul>
            {% endif %}
            
            <h3>Recommendations:</h3>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
            
            <p><em>Generated at {{ suite.end_time.strftime('%Y-%m-%d %H:%M:%S') }}</em></p>
            """,
            format="html",
            channels=["email", "slack"],
            conditions={"success_rate": {"min": 70, "max": 95}}
        )
        
        # Critical template
        critical_template = NotificationTemplate(
            name="test_critical",
            subject="🚨 CRITICAL: Test Suite Failed: {{ suite.suite_name }}",
            body="""
            <h2>CRITICAL: Test Suite Failed</h2>
            <p><strong>Suite:</strong> {{ suite.suite_name }}</p>
            <p><strong>Duration:</strong> {{ suite.total_duration|round(2) }} seconds</p>
            <p><strong>Results:</strong> {{ suite.passed }} passed, {{ suite.failed }} failed, {{ suite.skipped }} skipped</p>
            <p><strong>Success Rate:</strong> {{ suite.success_rate|round(1) }}%</p>
            <p><strong>Coverage:</strong> {{ suite.coverage_percentage|round(1) }}%</p>
            
            <h3>Failed Tests:</h3>
            <ul>
                {% for result in suite.results %}
                {% if result.status.value == 'failed' %}
                <li><strong>{{ result.test_name }}</strong> ({{ result.test_module }})
                    {% if result.error_message %}
                    <br><small>Error: {{ result.error_message|truncate(100) }}</small>
                    {% endif %}
                </li>
                {% endif %}
                {% endfor %}
            </ul>
            
            <h3>Immediate Actions Required:</h3>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
            
            <p><strong>⚠️ This requires immediate attention!</strong></p>
            <p><em>Generated at {{ suite.end_time.strftime('%Y-%m-%d %H:%M:%S') }}</em></p>
            """,
            format="html",
            channels=["email", "slack", "teams"],
            conditions={"success_rate": {"max": 70}}
        )
        
        # Store templates
        self.config.templates = [success_template, warning_template, critical_template]
    
    async def distribute_reports(self, 
                               suite: TestSuite,
                               reports: Dict[str, str],
                               coverage_report: Optional[CoverageReport] = None,
                               recommendations: List[str] = None) -> Dict[str, Any]:
        """Distribute test reports to configured channels"""
        
        if not self.config.enabled:
            self.logger.info("Report distribution is disabled")
            return {"status": "disabled"}
        
        distribution_id = self._generate_distribution_id(suite)
        
        # Determine notification level
        notification_level = self._determine_notification_level(suite)
        
        # Select appropriate template
        template = self._select_template(suite, notification_level)
        
        # Prepare notification data
        notification_data = {
            'suite': suite,
            'coverage': coverage_report,
            'recommendations': recommendations or [],
            'reports': reports,
            'notification_level': notification_level.value,
            'distribution_id': distribution_id
        }
        
        # Filter recipients based on level and schedule
        active_recipients = self._filter_recipients(notification_level)
        
        # Store initial distribution record
        await self._store_distribution_record(distribution_id, suite, active_recipients, reports)
        
        # Send notifications
        results = {}
        
        for recipient in active_recipients:
            recipient_results = await self._send_to_recipient(
                recipient, template, notification_data, reports
            )
            results[recipient.name] = recipient_results
        
        # Update distribution record
        await self._update_distribution_record(distribution_id, results)
        
        # Generate distribution summary
        summary = self._generate_distribution_summary(results)
        
        return {
            "distribution_id": distribution_id,
            "template_used": template.name,
            "notification_level": notification_level.value,
            "recipients_contacted": len(active_recipients),
            "results": results,
            "summary": summary
        }
    
    def _generate_distribution_id(self, suite: TestSuite) -> str:
        """Generate unique distribution ID"""
        data = f"{suite.suite_name}_{suite.end_time.isoformat()}_{suite.total_tests}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _determine_notification_level(self, suite: TestSuite) -> NotificationLevel:
        """Determine notification level based on test results"""
        if suite.success_rate < 70:
            return NotificationLevel.CRITICAL
        elif suite.success_rate < 95 or suite.failed > 0:
            return NotificationLevel.WARNING
        else:
            return NotificationLevel.INFO
    
    def _select_template(self, suite: TestSuite, level: NotificationLevel) -> NotificationTemplate:
        """Select appropriate template based on results and level"""
        
        # First, try to find template based on conditions
        for template in self.config.templates:
            if self._template_matches_conditions(template, suite):
                return template
        
        # Fallback to level-based selection
        if level == NotificationLevel.CRITICAL:
            return next((t for t in self.config.templates if t.name == "test_critical"), self.config.templates[0])
        elif level == NotificationLevel.WARNING:
            return next((t for t in self.config.templates if t.name == "test_warning"), self.config.templates[0])
        else:
            return next((t for t in self.config.templates if t.name == "test_success"), self.config.templates[0])
    
    def _template_matches_conditions(self, template: NotificationTemplate, suite: TestSuite) -> bool:
        """Check if template conditions match current suite results"""
        
        conditions = template.conditions
        
        # Check success rate conditions
        if 'success_rate' in conditions:
            sr_conditions = conditions['success_rate']
            if 'min' in sr_conditions and suite.success_rate < sr_conditions['min']:
                return False
            if 'max' in sr_conditions and suite.success_rate > sr_conditions['max']:
                return False
        
        # Check failure count conditions
        if 'max_failures' in conditions and suite.failed > conditions['max_failures']:
            return False
        
        # Check duration conditions
        if 'max_duration' in conditions and suite.total_duration > conditions['max_duration']:
            return False
        
        return True
    
    def _filter_recipients(self, level: NotificationLevel) -> List[NotificationRecipient]:
        """Filter recipients based on notification level and schedule"""
        
        active_recipients = []
        
        for recipient in self.config.recipients:
            # Check notification level
            if level.value not in recipient.notification_levels:
                continue
            
            # Check schedule
            if not self._is_recipient_available(recipient):
                continue
            
            active_recipients.append(recipient)
        
        return active_recipients
    
    def _is_recipient_available(self, recipient: NotificationRecipient) -> bool:
        """Check if recipient should receive notifications based on schedule"""
        
        schedule = recipient.schedule
        
        # Always notify
        if schedule.get("always", False):
            return True
        
        # Check time-based conditions
        now = datetime.now()
        
        # Check day of week
        if "days_of_week" in schedule:
            current_day = now.strftime("%A").lower()
            if current_day not in [day.lower() for day in schedule["days_of_week"]]:
                return False
        
        # Check time range
        if "time_range" in schedule:
            time_range = schedule["time_range"]
            start_time = datetime.strptime(time_range["start"], "%H:%M").time()
            end_time = datetime.strptime(time_range["end"], "%H:%M").time()
            current_time = now.time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        return True
    
    async def _send_to_recipient(self, 
                                recipient: NotificationRecipient,
                                template: NotificationTemplate,
                                data: Dict[str, Any],
                                reports: Dict[str, str]) -> Dict[str, Any]:
        """Send notification to a specific recipient"""
        
        results = {}
        
        # Filter reports based on recipient preferences
        filtered_reports = {
            format_name: path for format_name, path in reports.items()
            if format_name in recipient.report_formats
        }
        
        # Send via each configured channel
        for channel in template.channels:
            try:
                if channel == NotificationChannel.EMAIL.value and recipient.email:
                    result = await self._send_email(recipient, template, data, filtered_reports)
                elif channel == NotificationChannel.SLACK.value and recipient.slack_user_id:
                    result = await self._send_slack(recipient, template, data, filtered_reports)
                elif channel == NotificationChannel.TEAMS.value and recipient.teams_user_id:
                    result = await self._send_teams(recipient, template, data, filtered_reports)
                else:
                    result = {"status": "skipped", "reason": "channel not configured"}
                
                results[channel] = result
                
                # Log notification
                await self._log_notification(
                    data['distribution_id'], channel, recipient.name, 
                    result["status"], result.get("message", "")
                )
                
            except Exception as e:
                error_result = {"status": "error", "error": str(e)}
                results[channel] = error_result
                
                await self._log_notification(
                    data['distribution_id'], channel, recipient.name,
                    "error", str(e)
                )
        
        return results
    
    async def _send_email(self, 
                         recipient: NotificationRecipient,
                         template: NotificationTemplate,
                         data: Dict[str, Any],
                         reports: Dict[str, str]) -> Dict[str, Any]:
        """Send email notification"""
        
        email_config = self.config.email_config
        
        if not email_config.get("enabled", False):
            return {"status": "disabled", "reason": "email not configured"}
        
        try:
            # Render template
            jinja_template = self.jinja_env.from_string(template.body)
            body = jinja_template.render(**data)
            
            subject_template = self.jinja_env.from_string(template.subject)
            subject = subject_template.render(**data)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = recipient.email
            msg['Subject'] = subject
            
            # Add body
            if template.format == "html":
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Attach reports
            for format_name, report_path in reports.items():
                if Path(report_path).exists():
                    with open(report_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {Path(report_path).name}'
                        )
                        msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls(context=context)
                
                if email_config.get('username') and email_config.get('password'):
                    server.login(email_config['username'], email_config['password'])
                
                server.send_message(msg)
            
            return {"status": "success", "message": "Email sent successfully"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _send_slack(self, 
                         recipient: NotificationRecipient,
                         template: NotificationTemplate,
                         data: Dict[str, Any],
                         reports: Dict[str, str]) -> Dict[str, Any]:
        """Send Slack notification"""
        
        slack_config = self.config.slack_config
        
        if not slack_config.get("enabled", False):
            return {"status": "disabled", "reason": "slack not configured"}
        
        try:
            # Render template (convert HTML to Slack format)
            jinja_template = self.jinja_env.from_string(template.body)
            body = jinja_template.render(**data)
            
            # Convert HTML to Slack markdown (simplified)
            slack_body = self._convert_html_to_slack(body)
            
            subject_template = self.jinja_env.from_string(template.subject)
            subject = subject_template.render(**data)
            
            # Create Slack message
            message = {
                "channel": recipient.slack_user_id,
                "text": subject,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": slack_body
                        }
                    }
                ]
            }
            
            # Add attachment info
            if reports:
                attachment_text = "📎 Reports available: " + ", ".join(reports.keys())
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": attachment_text
                    }
                })
            
            # Send to Slack
            webhook_url = slack_config.get('webhook_url')
            if webhook_url:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=message) as response:
                        if response.status == 200:
                            return {"status": "success", "message": "Slack notification sent"}
                        else:
                            return {"status": "error", "error": f"Slack API returned {response.status}"}
            else:
                return {"status": "error", "error": "Slack webhook URL not configured"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _send_teams(self, 
                         recipient: NotificationRecipient,
                         template: NotificationTemplate,
                         data: Dict[str, Any],
                         reports: Dict[str, str]) -> Dict[str, Any]:
        """Send Microsoft Teams notification"""
        
        teams_config = self.config.teams_config
        
        if not teams_config.get("enabled", False):
            return {"status": "disabled", "reason": "teams not configured"}
        
        try:
            # Render template
            jinja_template = self.jinja_env.from_string(template.body)
            body = jinja_template.render(**data)
            
            subject_template = self.jinja_env.from_string(template.subject)
            subject = subject_template.render(**data)
            
            # Create Teams message
            message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": self._get_theme_color(data['notification_level']),
                "summary": subject,
                "sections": [
                    {
                        "activityTitle": subject,
                        "activityText": body,
                        "markdown": True
                    }
                ]
            }
            
            # Add facts
            facts = [
                {"name": "Suite", "value": data['suite'].suite_name},
                {"name": "Success Rate", "value": f"{data['suite'].success_rate:.1f}%"},
                {"name": "Duration", "value": f"{data['suite'].total_duration:.2f}s"},
                {"name": "Coverage", "value": f"{data['suite'].coverage_percentage:.1f}%"}
            ]
            
            message["sections"][0]["facts"] = facts
            
            # Send to Teams
            webhook_url = teams_config.get('webhook_url')
            if webhook_url:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=message) as response:
                        if response.status == 200:
                            return {"status": "success", "message": "Teams notification sent"}
                        else:
                            return {"status": "error", "error": f"Teams API returned {response.status}"}
            else:
                return {"status": "error", "error": "Teams webhook URL not configured"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _convert_html_to_slack(self, html_content: str) -> str:
        """Convert HTML to Slack markdown format"""
        
        # Simple HTML to Slack conversion
        content = html_content
        
        # Headers
        content = content.replace('<h2>', '*').replace('</h2>', '*\n')
        content = content.replace('<h3>', '*').replace('</h3>', '*\n')
        
        # Bold
        content = content.replace('<strong>', '*').replace('</strong>', '*')
        content = content.replace('<b>', '*').replace('</b>', '*')
        
        # Emphasis
        content = content.replace('<em>', '_').replace('</em>', '_')
        content = content.replace('<i>', '_').replace('</i>', '_')
        
        # Lists
        content = content.replace('<ul>', '').replace('</ul>', '')
        content = content.replace('<li>', '• ').replace('</li>', '\n')
        
        # Paragraphs
        content = content.replace('<p>', '').replace('</p>', '\n')
        
        # Remove remaining HTML tags
        import re
        content = re.sub(r'<[^>]+>', '', content)
        
        return content.strip()
    
    def _get_theme_color(self, level: str) -> str:
        """Get theme color for Teams message"""
        color_map = {
            "info": "00FF00",      # Green
            "warning": "FFFF00",   # Yellow
            "error": "FF6600",     # Orange
            "critical": "FF0000"   # Red
        }
        return color_map.get(level, "0078D4")  # Default blue
    
    async def _store_distribution_record(self, 
                                       distribution_id: str,
                                       suite: TestSuite,
                                       recipients: List[NotificationRecipient],
                                       reports: Dict[str, str]):
        """Store distribution record in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO distribution_history 
            (distribution_id, suite_name, generated_at, channels, recipients, 
             status, reports_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            distribution_id,
            suite.suite_name,
            suite.end_time,
            json.dumps([channel.value for channel in NotificationChannel]),
            json.dumps([recipient.name for recipient in recipients]),
            "in_progress",
            json.dumps(list(reports.keys()))
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_distribution_record(self, 
                                        distribution_id: str,
                                        results: Dict[str, Any]):
        """Update distribution record with results"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine overall status
        all_successful = all(
            all(channel_result.get("status") == "success" for channel_result in recipient_results.values())
            for recipient_results in results.values()
        )
        
        status = "completed" if all_successful else "partial_failure"
        
        cursor.execute('''
            UPDATE distribution_history 
            SET status = ?, error_message = ?
            WHERE distribution_id = ?
        ''', (status, json.dumps(results), distribution_id))
        
        conn.commit()
        conn.close()
    
    async def _log_notification(self, 
                              distribution_id: str,
                              channel: str,
                              recipient: str,
                              status: str,
                              message: str):
        """Log individual notification attempt"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notification_log 
            (distribution_id, channel, recipient, status, message, sent_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (distribution_id, channel, recipient, status, message, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def _generate_distribution_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of distribution results"""
        
        total_recipients = len(results)
        successful_notifications = 0
        failed_notifications = 0
        
        for recipient_results in results.values():
            for channel_result in recipient_results.values():
                if channel_result.get("status") == "success":
                    successful_notifications += 1
                else:
                    failed_notifications += 1
        
        return {
            "total_recipients": total_recipients,
            "successful_notifications": successful_notifications,
            "failed_notifications": failed_notifications,
            "success_rate": (successful_notifications / (successful_notifications + failed_notifications) * 100) if (successful_notifications + failed_notifications) > 0 else 0,
            "channels_used": list(set(
                channel for recipient_results in results.values() 
                for channel in recipient_results.keys()
            ))
        }
    
    async def get_distribution_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get distribution history"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM distribution_history 
            WHERE created_at > datetime('now', '-{} days')
            ORDER BY created_at DESC
        '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    async def get_notification_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get notification statistics"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get distribution stats
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_distributions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_distributions,
                AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
            FROM distribution_history 
            WHERE created_at > datetime('now', '-{} days')
        '''.format(days))
        
        dist_stats = cursor.fetchone()
        
        # Get channel stats
        cursor.execute('''
            SELECT 
                channel,
                COUNT(*) as total_notifications,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_notifications,
                AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
            FROM notification_log 
            WHERE created_at > datetime('now', '-{} days')
            GROUP BY channel
        '''.format(days))
        
        channel_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "distribution_stats": {
                "total_distributions": dist_stats[0] or 0,
                "successful_distributions": dist_stats[1] or 0,
                "success_rate": dist_stats[2] or 0
            },
            "channel_stats": [
                {
                    "channel": row[0],
                    "total_notifications": row[1],
                    "successful_notifications": row[2],
                    "success_rate": row[3]
                }
                for row in channel_stats
            ]
        }
    
    async def retry_failed_distributions(self) -> Dict[str, Any]:
        """Retry failed distributions"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get failed distributions
        cursor.execute('''
            SELECT * FROM distribution_history 
            WHERE status = 'partial_failure' 
            AND retry_count < ?
            AND created_at > datetime('now', '-1 days')
        ''', (self.config.retry_config['max_retries'],))
        
        failed_distributions = cursor.fetchall()
        conn.close()
        
        retry_results = {}
        
        for dist_record in failed_distributions:
            distribution_id = dist_record[1]  # Assuming distribution_id is second column
            
            # Increment retry count
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE distribution_history 
                SET retry_count = retry_count + 1 
                WHERE distribution_id = ?
            ''', (distribution_id,))
            conn.commit()
            conn.close()
            
            # Wait before retry
            await asyncio.sleep(self.config.retry_config['delay_seconds'])
            
            # This would need to be implemented with proper retry logic
            # For now, just log the attempt
            retry_results[distribution_id] = {"status": "retry_logged"}
        
        return {
            "retried_distributions": len(failed_distributions),
            "results": retry_results
        }
    
    def add_recipient(self, recipient: NotificationRecipient):
        """Add a new recipient"""
        self.config.recipients.append(recipient)
    
    def remove_recipient(self, recipient_name: str):
        """Remove a recipient"""
        self.config.recipients = [
            r for r in self.config.recipients 
            if r.name != recipient_name
        ]
    
    def add_template(self, template: NotificationTemplate):
        """Add a new template"""
        self.config.templates.append(template)
    
    def update_template(self, template_name: str, template: NotificationTemplate):
        """Update an existing template"""
        for i, t in enumerate(self.config.templates):
            if t.name == template_name:
                self.config.templates[i] = template
                break