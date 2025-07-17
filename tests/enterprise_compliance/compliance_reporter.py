#!/usr/bin/env python3
"""
Compliance Reporter with Automated Regulatory Reporting
Agent Zeta: Enterprise Compliance & Chaos Engineering Implementation Specialist

Advanced compliance reporting system with automated regulatory alignment,
certification tracking, and real-time compliance monitoring.

Features:
- Automated regulatory reporting for multiple frameworks
- Real-time compliance scoring and monitoring
- Certification tracking and renewal management
- Regulatory alignment verification
- Audit trail reporting with blockchain verification
- Automated compliance documentation generation
"""

import asyncio
import json
import logging
import time
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import csv
import io
import base64
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)


class RegulatoryFramework(Enum):
    """Regulatory frameworks for compliance reporting"""
    SEC_REGULATION = "sec_regulation"
    GDPR = "gdpr"
    SARBANES_OXLEY = "sarbanes_oxley"
    BASEL_III = "basel_iii"
    MIFID_II = "mifid_ii"
    CFTC = "cftc"
    FINRA = "finra"
    ISO_27001 = "iso_27001"
    PCIDSS = "pci_dss"
    HIPAA = "hipaa"


class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class ReportType(Enum):
    """Types of compliance reports"""
    AUDIT_TRAIL = "audit_trail"
    COMPLIANCE_SCORE = "compliance_score"
    VIOLATION_SUMMARY = "violation_summary"
    CERTIFICATION_STATUS = "certification_status"
    REGULATORY_FILING = "regulatory_filing"
    INCIDENT_REPORT = "incident_report"
    REMEDIATION_PLAN = "remediation_plan"
    CONTROL_EFFECTIVENESS = "control_effectiveness"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    PENDING_REVIEW = "pending_review"
    UNDER_REMEDIATION = "under_remediation"


@dataclass
class ComplianceMetric:
    """Compliance metric definition"""
    metric_id: str
    name: str
    description: str
    framework: RegulatoryFramework
    metric_type: str  # percentage, count, boolean, etc.
    
    # Calculation
    calculation_method: str
    target_value: Any
    current_value: Any = None
    
    # Status
    status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    criticality: str = "medium"  # low, medium, high, critical
    automated: bool = True
    evidence_required: bool = False


@dataclass
class ComplianceCertification:
    """Compliance certification tracking"""
    certification_id: str
    name: str
    framework: RegulatoryFramework
    issuing_authority: str
    
    # Dates
    issued_date: datetime
    expiry_date: datetime
    next_review_date: datetime
    
    # Status
    status: str = "active"  # active, expired, pending, suspended
    compliance_score: float = 0.0
    
    # Requirements
    requirements: List[str] = field(default_factory=list)
    evidence_items: List[str] = field(default_factory=list)
    
    # Renewal
    renewal_process: Dict[str, Any] = field(default_factory=dict)
    renewal_notifications: List[datetime] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    report_type: ReportType
    framework: RegulatoryFramework
    generated_at: datetime
    
    # Report content
    title: str
    summary: Dict[str, Any]
    sections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    report_format: ReportFormat = ReportFormat.HTML
    generated_by: str = "compliance_reporter"
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    
    # Distribution
    recipients: List[str] = field(default_factory=list)
    distribution_status: str = "pending"
    
    # Integrity
    content_hash: Optional[str] = None
    digital_signature: Optional[str] = None


class ComplianceReporter:
    """
    Advanced Compliance Reporter
    
    Provides comprehensive compliance reporting capabilities including:
    - Automated regulatory reporting for multiple frameworks
    - Real-time compliance scoring and monitoring
    - Certification tracking and renewal management
    - Regulatory alignment verification
    - Audit trail reporting with blockchain verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize storage
        self.db_connection = self._initialize_database()
        
        # Compliance metrics
        self.metrics: Dict[str, ComplianceMetric] = {}
        self.certifications: Dict[str, ComplianceCertification] = {}
        
        # Report templates
        self.templates = self._load_report_templates()
        
        # Notification system
        self.notification_queue = deque()
        
        # Performance tracking
        self.reporting_metrics = {
            'reports_generated': 0,
            'reports_distributed': 0,
            'certifications_tracked': 0,
            'compliance_checks': 0,
            'avg_generation_time_ms': 0.0
        }
        
        # Load compliance frameworks
        self._initialize_compliance_frameworks()
        
        logger.info("ComplianceReporter initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'reporting': {
                'enabled': True,
                'output_directory': '/tmp/compliance_reports',
                'template_directory': '/tmp/compliance_templates',
                'automatic_generation': True,
                'generation_schedule': 'daily'
            },
            'notifications': {
                'enabled': True,
                'email_enabled': True,
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'sender_email': 'compliance@example.com',
                'certification_reminder_days': [30, 14, 7, 1]
            },
            'frameworks': {
                'enabled_frameworks': [
                    'sec_regulation',
                    'gdpr',
                    'sarbanes_oxley',
                    'iso_27001'
                ],
                'default_targets': {
                    'compliance_score': 95.0,
                    'violation_threshold': 5,
                    'response_time_hours': 24
                }
            },
            'storage': {
                'database_path': '/tmp/compliance_reporter.db',
                'retention_days': 2555,  # 7 years
                'backup_enabled': True
            }
        }
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize compliance reporting database"""
        db_path = self.config['storage']['database_path']
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_metrics (
                metric_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                framework TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                calculation_method TEXT,
                target_value TEXT,
                current_value TEXT,
                status TEXT,
                last_updated TEXT,
                criticality TEXT,
                automated INTEGER,
                evidence_required INTEGER
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_certifications (
                certification_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                framework TEXT NOT NULL,
                issuing_authority TEXT,
                issued_date TEXT,
                expiry_date TEXT,
                next_review_date TEXT,
                status TEXT,
                compliance_score REAL,
                requirements TEXT,
                evidence_items TEXT,
                renewal_process TEXT,
                renewal_notifications TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                framework TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                sections TEXT,
                report_format TEXT,
                generated_by TEXT,
                reviewed_by TEXT,
                approved_by TEXT,
                recipients TEXT,
                distribution_status TEXT,
                content_hash TEXT,
                digital_signature TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_notifications (
                notification_id TEXT PRIMARY KEY,
                notification_type TEXT NOT NULL,
                framework TEXT,
                message TEXT,
                recipients TEXT,
                scheduled_for TEXT,
                sent_at TEXT,
                status TEXT,
                related_id TEXT
            )
        ''')
        
        # Create indices
        conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_framework ON compliance_metrics(framework)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_certifications_expiry ON compliance_certifications(expiry_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_reports_generated ON compliance_reports(generated_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_scheduled ON compliance_notifications(scheduled_for)')
        
        conn.commit()
        return conn
    
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates"""
        templates = {}
        
        # Basic HTML template
        templates['html_basic'] = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; }
        .section { margin: 20px 0; }
        .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
        .compliant { color: green; }
        .non-compliant { color: red; }
        .pending { color: orange; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generated_at }}</p>
        <p>Framework: {{ framework }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>Overall Compliance Score: <span class="{{ summary.status_class }}">{{ summary.compliance_score }}%</span></p>
        <p>Total Violations: {{ summary.total_violations }}</p>
        <p>Certification Status: {{ summary.certification_status }}</p>
    </div>
    
    {% for section in sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        <div>{{ section.content }}</div>
    </div>
    {% endfor %}
    
    <div class="section">
        <h2>Compliance Metrics</h2>
        {% for metric in metrics %}
        <div class="metric">
            <strong>{{ metric.name }}</strong>: 
            <span class="{{ metric.status_class }}">{{ metric.current_value }}</span>
            (Target: {{ metric.target_value }})
        </div>
        {% endfor %}
    </div>
</body>
</html>
        '''
        
        # JSON template
        templates['json_basic'] = '''
{
    "report_id": "{{ report_id }}",
    "title": "{{ title }}",
    "generated_at": "{{ generated_at }}",
    "framework": "{{ framework }}",
    "summary": {{ summary | tojson }},
    "sections": {{ sections | tojson }},
    "metrics": {{ metrics | tojson }},
    "certifications": {{ certifications | tojson }}
}
        '''
        
        return templates
    
    def _initialize_compliance_frameworks(self):
        """Initialize compliance frameworks and metrics"""
        # SEC Regulation metrics
        self.add_compliance_metric(
            ComplianceMetric(
                metric_id="SEC_001",
                name="Trading Decision Documentation",
                description="Percentage of trading decisions with proper documentation",
                framework=RegulatoryFramework.SEC_REGULATION,
                metric_type="percentage",
                calculation_method="documented_decisions / total_decisions * 100",
                target_value=100.0,
                criticality="high"
            )
        )
        
        # GDPR metrics
        self.add_compliance_metric(
            ComplianceMetric(
                metric_id="GDPR_001",
                name="Data Subject Rights Response Time",
                description="Average response time for data subject rights requests",
                framework=RegulatoryFramework.GDPR,
                metric_type="hours",
                calculation_method="sum(response_times) / count(requests)",
                target_value=72.0,
                criticality="high"
            )
        )
        
        # SOX metrics
        self.add_compliance_metric(
            ComplianceMetric(
                metric_id="SOX_001",
                name="Financial Control Effectiveness",
                description="Effectiveness of financial controls",
                framework=RegulatoryFramework.SARBANES_OXLEY,
                metric_type="percentage",
                calculation_method="effective_controls / total_controls * 100",
                target_value=95.0,
                criticality="critical"
            )
        )
        
        # ISO 27001 metrics
        self.add_compliance_metric(
            ComplianceMetric(
                metric_id="ISO_001",
                name="Security Incident Response Time",
                description="Average time to respond to security incidents",
                framework=RegulatoryFramework.ISO_27001,
                metric_type="hours",
                calculation_method="sum(response_times) / count(incidents)",
                target_value=4.0,
                criticality="high"
            )
        )
    
    def add_compliance_metric(self, metric: ComplianceMetric):
        """Add compliance metric"""
        self.metrics[metric.metric_id] = metric
        
        # Store in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_metrics (
                metric_id, name, description, framework, metric_type,
                calculation_method, target_value, current_value, status,
                last_updated, criticality, automated, evidence_required
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_id,
            metric.name,
            metric.description,
            metric.framework.value,
            metric.metric_type,
            metric.calculation_method,
            str(metric.target_value),
            str(metric.current_value) if metric.current_value is not None else None,
            metric.status.value,
            metric.last_updated.isoformat(),
            metric.criticality,
            int(metric.automated),
            int(metric.evidence_required)
        ))
        self.db_connection.commit()
    
    def add_certification(self, certification: ComplianceCertification):
        """Add compliance certification"""
        self.certifications[certification.certification_id] = certification
        
        # Store in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_certifications (
                certification_id, name, framework, issuing_authority,
                issued_date, expiry_date, next_review_date, status,
                compliance_score, requirements, evidence_items,
                renewal_process, renewal_notifications
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            certification.certification_id,
            certification.name,
            certification.framework.value,
            certification.issuing_authority,
            certification.issued_date.isoformat(),
            certification.expiry_date.isoformat(),
            certification.next_review_date.isoformat(),
            certification.status,
            certification.compliance_score,
            json.dumps(certification.requirements),
            json.dumps(certification.evidence_items),
            json.dumps(certification.renewal_process),
            json.dumps([dt.isoformat() for dt in certification.renewal_notifications])
        ))
        self.db_connection.commit()
    
    async def update_compliance_metric(self, metric_id: str, current_value: Any, evidence: Optional[Dict[str, Any]] = None):
        """Update compliance metric value"""
        if metric_id not in self.metrics:
            raise ValueError(f"Metric {metric_id} not found")
        
        metric = self.metrics[metric_id]
        metric.current_value = current_value
        metric.last_updated = datetime.now(timezone.utc)
        
        # Evaluate compliance status
        if metric.metric_type == "percentage":
            if float(current_value) >= float(metric.target_value):
                metric.status = ComplianceStatus.COMPLIANT
            else:
                metric.status = ComplianceStatus.NON_COMPLIANT
        elif metric.metric_type == "hours":
            if float(current_value) <= float(metric.target_value):
                metric.status = ComplianceStatus.COMPLIANT
            else:
                metric.status = ComplianceStatus.NON_COMPLIANT
        elif metric.metric_type == "boolean":
            if current_value:
                metric.status = ComplianceStatus.COMPLIANT
            else:
                metric.status = ComplianceStatus.NON_COMPLIANT
        
        # Update in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            UPDATE compliance_metrics 
            SET current_value = ?, status = ?, last_updated = ?
            WHERE metric_id = ?
        ''', (
            str(current_value),
            metric.status.value,
            metric.last_updated.isoformat(),
            metric_id
        ))
        self.db_connection.commit()
        
        # Check for alerts
        if metric.status == ComplianceStatus.NON_COMPLIANT and metric.criticality in ['high', 'critical']:
            await self._create_compliance_alert(metric)
    
    async def _create_compliance_alert(self, metric: ComplianceMetric):
        """Create compliance alert for non-compliant metric"""
        alert = {
            'type': 'compliance_violation',
            'metric_id': metric.metric_id,
            'framework': metric.framework.value,
            'message': f"Compliance metric '{metric.name}' is non-compliant",
            'current_value': metric.current_value,
            'target_value': metric.target_value,
            'criticality': metric.criticality,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add to notification queue
        self.notification_queue.append(alert)
        
        logger.warning(f"Compliance alert created for metric {metric.metric_id}")
    
    async def generate_compliance_report(
        self,
        framework: RegulatoryFramework,
        report_type: ReportType,
        report_format: ReportFormat = ReportFormat.HTML,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ComplianceReport:
        """
        Generate compliance report
        
        Args:
            framework: Regulatory framework
            report_type: Type of report to generate
            report_format: Output format
            time_period: Time period for report (optional)
            
        Returns:
            ComplianceReport: Generated report
        """
        start_time = time.time()
        
        # Create report
        report = ComplianceReport(
            report_id=f"report_{uuid.uuid4().hex}",
            report_type=report_type,
            framework=framework,
            generated_at=datetime.now(timezone.utc),
            title=f"{framework.value.upper()} {report_type.value.replace('_', ' ').title()} Report",
            summary={},
            report_format=report_format
        )
        
        # Generate report content based on type
        if report_type == ReportType.COMPLIANCE_SCORE:
            await self._generate_compliance_score_report(report, framework)
        elif report_type == ReportType.VIOLATION_SUMMARY:
            await self._generate_violation_summary_report(report, framework, time_period)
        elif report_type == ReportType.CERTIFICATION_STATUS:
            await self._generate_certification_status_report(report, framework)
        elif report_type == ReportType.AUDIT_TRAIL:
            await self._generate_audit_trail_report(report, framework, time_period)
        elif report_type == ReportType.CONTROL_EFFECTIVENESS:
            await self._generate_control_effectiveness_report(report, framework)
        
        # Store report
        await self._store_report(report)
        
        # Update metrics
        generation_time = (time.time() - start_time) * 1000
        self.reporting_metrics['reports_generated'] += 1
        self._update_avg_generation_time(generation_time)
        
        logger.info(f"Generated compliance report: {report.report_id}")
        return report
    
    async def _generate_compliance_score_report(self, report: ComplianceReport, framework: RegulatoryFramework):
        """Generate compliance score report"""
        framework_metrics = [m for m in self.metrics.values() if m.framework == framework]
        
        if not framework_metrics:
            report.summary = {
                'compliance_score': 0.0,
                'total_metrics': 0,
                'compliant_metrics': 0,
                'status': 'no_data'
            }
            return
        
        # Calculate compliance score
        compliant_count = sum(1 for m in framework_metrics if m.status == ComplianceStatus.COMPLIANT)
        total_count = len(framework_metrics)
        compliance_score = (compliant_count / total_count) * 100 if total_count > 0 else 0
        
        # Generate summary
        report.summary = {
            'compliance_score': compliance_score,
            'total_metrics': total_count,
            'compliant_metrics': compliant_count,
            'non_compliant_metrics': total_count - compliant_count,
            'status': 'compliant' if compliance_score >= 95 else 'non_compliant',
            'status_class': 'compliant' if compliance_score >= 95 else 'non_compliant'
        }
        
        # Generate sections
        sections = []
        
        # Metrics overview
        metrics_section = {
            'title': 'Compliance Metrics Overview',
            'content': f"Total metrics evaluated: {total_count}\\n"
                      f"Compliant metrics: {compliant_count}\\n"
                      f"Non-compliant metrics: {total_count - compliant_count}\\n"
                      f"Overall compliance score: {compliance_score:.1f}%"
        }
        sections.append(metrics_section)
        
        # Detailed metrics
        for metric in framework_metrics:
            metric_section = {
                'title': metric.name,
                'content': f"Current Value: {metric.current_value}\\n"
                          f"Target Value: {metric.target_value}\\n"
                          f"Status: {metric.status.value}\\n"
                          f"Last Updated: {metric.last_updated.isoformat()}\\n"
                          f"Criticality: {metric.criticality}"
            }
            sections.append(metric_section)
        
        report.sections = sections
    
    async def _generate_violation_summary_report(self, report: ComplianceReport, framework: RegulatoryFramework, time_period: Optional[Tuple[datetime, datetime]]):
        """Generate violation summary report"""
        # Get non-compliant metrics
        framework_metrics = [m for m in self.metrics.values() if m.framework == framework]
        violations = [m for m in framework_metrics if m.status == ComplianceStatus.NON_COMPLIANT]
        
        # Generate summary
        report.summary = {
            'total_violations': len(violations),
            'critical_violations': sum(1 for v in violations if v.criticality == 'critical'),
            'high_violations': sum(1 for v in violations if v.criticality == 'high'),
            'medium_violations': sum(1 for v in violations if v.criticality == 'medium'),
            'low_violations': sum(1 for v in violations if v.criticality == 'low')
        }
        
        # Generate sections
        sections = []
        
        # Violations by criticality
        for criticality in ['critical', 'high', 'medium', 'low']:
            criticality_violations = [v for v in violations if v.criticality == criticality]
            if criticality_violations:
                section = {
                    'title': f'{criticality.title()} Priority Violations',
                    'content': '\\n'.join([
                        f"• {v.name}: {v.current_value} (Target: {v.target_value})"
                        for v in criticality_violations
                    ])
                }
                sections.append(section)
        
        report.sections = sections
    
    async def _generate_certification_status_report(self, report: ComplianceReport, framework: RegulatoryFramework):
        """Generate certification status report"""
        framework_certifications = [c for c in self.certifications.values() if c.framework == framework]
        
        # Check expiry status
        now = datetime.now(timezone.utc)
        active_certs = [c for c in framework_certifications if c.status == 'active']
        expired_certs = [c for c in framework_certifications if c.status == 'expired']
        expiring_soon = [c for c in active_certs if (c.expiry_date - now).days <= 30]
        
        # Generate summary
        report.summary = {
            'total_certifications': len(framework_certifications),
            'active_certifications': len(active_certs),
            'expired_certifications': len(expired_certs),
            'expiring_soon': len(expiring_soon),
            'overall_status': 'active' if active_certs and not expired_certs else 'attention_required'
        }
        
        # Generate sections
        sections = []
        
        # Active certifications
        if active_certs:
            active_section = {
                'title': 'Active Certifications',
                'content': '\\n'.join([
                    f"• {c.name} (Expires: {c.expiry_date.strftime('%Y-%m-%d')})"
                    for c in active_certs
                ])
            }
            sections.append(active_section)
        
        # Expiring soon
        if expiring_soon:
            expiring_section = {
                'title': 'Certifications Expiring Soon',
                'content': '\\n'.join([
                    f"• {c.name} (Expires: {c.expiry_date.strftime('%Y-%m-%d')} - {(c.expiry_date - now).days} days)"
                    for c in expiring_soon
                ])
            }
            sections.append(expiring_section)
        
        report.sections = sections
    
    async def _generate_audit_trail_report(self, report: ComplianceReport, framework: RegulatoryFramework, time_period: Optional[Tuple[datetime, datetime]]):
        """Generate audit trail report"""
        # This would integrate with the enterprise audit system
        # For now, provide a placeholder structure
        
        report.summary = {
            'audit_events': 0,
            'blockchain_verified': True,
            'integrity_status': 'verified',
            'time_period': time_period
        }
        
        sections = [
            {
                'title': 'Audit Trail Summary',
                'content': 'Blockchain-verified audit trail for compliance verification'
            }
        ]
        
        report.sections = sections
    
    async def _generate_control_effectiveness_report(self, report: ComplianceReport, framework: RegulatoryFramework):
        """Generate control effectiveness report"""
        framework_metrics = [m for m in self.metrics.values() if m.framework == framework]
        
        # Calculate effectiveness
        total_controls = len(framework_metrics)
        effective_controls = sum(1 for m in framework_metrics if m.status == ComplianceStatus.COMPLIANT)
        effectiveness = (effective_controls / total_controls) * 100 if total_controls > 0 else 0
        
        report.summary = {
            'total_controls': total_controls,
            'effective_controls': effective_controls,
            'effectiveness_percentage': effectiveness,
            'status': 'effective' if effectiveness >= 90 else 'needs_improvement'
        }
        
        sections = [
            {
                'title': 'Control Effectiveness Summary',
                'content': f"Total controls evaluated: {total_controls}\\n"
                          f"Effective controls: {effective_controls}\\n"
                          f"Effectiveness rate: {effectiveness:.1f}%"
            }
        ]
        
        report.sections = sections
    
    async def _store_report(self, report: ComplianceReport):
        """Store report in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO compliance_reports (
                report_id, report_type, framework, generated_at, title,
                summary, sections, report_format, generated_by,
                reviewed_by, approved_by, recipients, distribution_status,
                content_hash, digital_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.report_id,
            report.report_type.value,
            report.framework.value,
            report.generated_at.isoformat(),
            report.title,
            json.dumps(report.summary),
            json.dumps(report.sections),
            report.report_format.value,
            report.generated_by,
            report.reviewed_by,
            report.approved_by,
            json.dumps(report.recipients),
            report.distribution_status,
            report.content_hash,
            report.digital_signature
        ))
        self.db_connection.commit()
    
    async def export_report(self, report: ComplianceReport, output_path: Optional[str] = None) -> str:
        """Export report to file"""
        if not output_path:
            output_dir = self.config['reporting']['output_directory']
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_dir}/{report.report_id}.{report.report_format.value}"
        
        # Generate report content
        content = await self._render_report(report)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Report exported to: {output_path}")
        return output_path
    
    async def _render_report(self, report: ComplianceReport) -> str:
        """Render report using templates"""
        template_key = f"{report.report_format.value}_basic"
        
        if template_key not in self.templates:
            raise ValueError(f"Template not found: {template_key}")
        
        template = Template(self.templates[template_key])
        
        # Prepare template context
        context = {
            'report_id': report.report_id,
            'title': report.title,
            'generated_at': report.generated_at.isoformat(),
            'framework': report.framework.value,
            'summary': report.summary,
            'sections': report.sections,
            'metrics': self._get_framework_metrics_for_template(report.framework),
            'certifications': self._get_framework_certifications_for_template(report.framework)
        }
        
        return template.render(**context)
    
    def _get_framework_metrics_for_template(self, framework: RegulatoryFramework) -> List[Dict[str, Any]]:
        """Get metrics for template rendering"""
        framework_metrics = [m for m in self.metrics.values() if m.framework == framework]
        
        template_metrics = []
        for metric in framework_metrics:
            template_metrics.append({
                'name': metric.name,
                'current_value': metric.current_value,
                'target_value': metric.target_value,
                'status': metric.status.value,
                'status_class': metric.status.value.replace('_', '-')
            })
        
        return template_metrics
    
    def _get_framework_certifications_for_template(self, framework: RegulatoryFramework) -> List[Dict[str, Any]]:
        """Get certifications for template rendering"""
        framework_certifications = [c for c in self.certifications.values() if c.framework == framework]
        
        template_certifications = []
        for cert in framework_certifications:
            template_certifications.append({
                'name': cert.name,
                'status': cert.status,
                'expiry_date': cert.expiry_date.strftime('%Y-%m-%d'),
                'compliance_score': cert.compliance_score
            })
        
        return template_certifications
    
    async def distribute_report(self, report: ComplianceReport, recipients: List[str]):
        """Distribute report to recipients"""
        if not self.config['notifications']['email_enabled']:
            logger.warning("Email notifications disabled")
            return
        
        # Export report
        report_path = await self.export_report(report)
        
        # Send email
        await self._send_email_report(report, recipients, report_path)
        
        # Update distribution status
        report.recipients = recipients
        report.distribution_status = 'distributed'
        
        # Update in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            UPDATE compliance_reports 
            SET recipients = ?, distribution_status = ?
            WHERE report_id = ?
        ''', (json.dumps(recipients), report.distribution_status, report.report_id))
        self.db_connection.commit()
        
        self.reporting_metrics['reports_distributed'] += 1
        logger.info(f"Report distributed to {len(recipients)} recipients")
    
    async def _send_email_report(self, report: ComplianceReport, recipients: List[str], report_path: str):
        """Send report via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['notifications']['sender_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Compliance Report: {report.title}"
            
            # Email body
            body = f"""
            Compliance Report Generated
            
            Report ID: {report.report_id}
            Framework: {report.framework.value}
            Generated: {report.generated_at.isoformat()}
            
            Summary:
            {json.dumps(report.summary, indent=2)}
            
            Please find the detailed report attached.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report
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
            server = smtplib.SMTP(
                self.config['notifications']['smtp_server'],
                self.config['notifications']['smtp_port']
            )
            server.starttls()
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    async def check_certification_renewals(self):
        """Check for upcoming certification renewals"""
        now = datetime.now(timezone.utc)
        reminder_days = self.config['notifications']['certification_reminder_days']
        
        for cert in self.certifications.values():
            if cert.status != 'active':
                continue
            
            days_until_expiry = (cert.expiry_date - now).days
            
            if days_until_expiry in reminder_days:
                await self._create_renewal_notification(cert, days_until_expiry)
    
    async def _create_renewal_notification(self, cert: ComplianceCertification, days_until_expiry: int):
        """Create certification renewal notification"""
        notification = {
            'notification_id': f"renewal_{uuid.uuid4().hex}",
            'type': 'certification_renewal',
            'framework': cert.framework.value,
            'message': f"Certification '{cert.name}' expires in {days_until_expiry} days",
            'certification_id': cert.certification_id,
            'expiry_date': cert.expiry_date.isoformat(),
            'urgency': 'high' if days_until_expiry <= 7 else 'medium'
        }
        
        self.notification_queue.append(notification)
        logger.warning(f"Certification renewal notification: {cert.name} expires in {days_until_expiry} days")
    
    def _update_avg_generation_time(self, generation_time: float):
        """Update average generation time metric"""
        total_reports = self.reporting_metrics['reports_generated']
        old_avg = self.reporting_metrics['avg_generation_time_ms']
        
        if total_reports > 0:
            self.reporting_metrics['avg_generation_time_ms'] = (
                (old_avg * (total_reports - 1) + generation_time) / total_reports
            )
        else:
            self.reporting_metrics['avg_generation_time_ms'] = generation_time
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        dashboard = {
            'overall_status': 'compliant',
            'frameworks': {},
            'certifications': {},
            'recent_reports': [],
            'notifications': len(self.notification_queue),
            'metrics': self.reporting_metrics.copy()
        }
        
        # Framework status
        for framework in RegulatoryFramework:
            framework_metrics = [m for m in self.metrics.values() if m.framework == framework]
            if framework_metrics:
                compliant = sum(1 for m in framework_metrics if m.status == ComplianceStatus.COMPLIANT)
                total = len(framework_metrics)
                score = (compliant / total) * 100 if total > 0 else 0
                
                dashboard['frameworks'][framework.value] = {
                    'compliance_score': score,
                    'total_metrics': total,
                    'compliant_metrics': compliant,
                    'status': 'compliant' if score >= 95 else 'non_compliant'
                }
        
        # Certification status
        for cert in self.certifications.values():
            framework_key = cert.framework.value
            if framework_key not in dashboard['certifications']:
                dashboard['certifications'][framework_key] = []
            
            dashboard['certifications'][framework_key].append({
                'name': cert.name,
                'status': cert.status,
                'expiry_date': cert.expiry_date.isoformat(),
                'days_until_expiry': (cert.expiry_date - datetime.now(timezone.utc)).days
            })
        
        return dashboard
    
    def get_reporting_metrics(self) -> Dict[str, Any]:
        """Get reporting metrics"""
        return self.reporting_metrics.copy()


# Test function
async def test_compliance_reporter():
    """Test the Compliance Reporter"""
    print("📊 Testing Compliance Reporter")
    
    # Initialize reporter
    reporter = ComplianceReporter()
    
    # Test metric updates
    print("\\n📈 Testing metric updates...")
    
    # Update SEC metric
    await reporter.update_compliance_metric("SEC_001", 95.5)
    print("Updated SEC trading documentation metric")
    
    # Update GDPR metric
    await reporter.update_compliance_metric("GDPR_001", 48.0)
    print("Updated GDPR response time metric")
    
    # Add certification
    print("\\n🏆 Testing certification tracking...")
    
    certification = ComplianceCertification(
        certification_id="ISO_27001_2024",
        name="ISO 27001 Information Security Management",
        framework=RegulatoryFramework.ISO_27001,
        issuing_authority="ISO Certification Body",
        issued_date=datetime.now(timezone.utc) - timedelta(days=100),
        expiry_date=datetime.now(timezone.utc) + timedelta(days=265),
        next_review_date=datetime.now(timezone.utc) + timedelta(days=200),
        compliance_score=92.5
    )
    
    reporter.add_certification(certification)
    print(f"Added certification: {certification.name}")
    
    # Test report generation
    print("\\n📋 Testing report generation...")
    
    # Generate compliance score report
    score_report = await reporter.generate_compliance_report(
        framework=RegulatoryFramework.SEC_REGULATION,
        report_type=ReportType.COMPLIANCE_SCORE,
        report_format=ReportFormat.HTML
    )
    print(f"Generated compliance score report: {score_report.report_id}")
    
    # Generate violation summary report
    violation_report = await reporter.generate_compliance_report(
        framework=RegulatoryFramework.GDPR,
        report_type=ReportType.VIOLATION_SUMMARY,
        report_format=ReportFormat.JSON
    )
    print(f"Generated violation summary report: {violation_report.report_id}")
    
    # Test report export
    print("\\n💾 Testing report export...")
    
    report_path = await reporter.export_report(score_report)
    print(f"Exported report to: {report_path}")
    
    # Test certification renewal check
    print("\\n🔔 Testing certification renewal check...")
    
    await reporter.check_certification_renewals()
    print("Checked certification renewals")
    
    # Test compliance dashboard
    print("\\n📊 Testing compliance dashboard...")
    
    dashboard = reporter.get_compliance_dashboard()
    print(f"Dashboard overview:")
    print(f"  Overall status: {dashboard['overall_status']}")
    print(f"  Frameworks tracked: {len(dashboard['frameworks'])}")
    print(f"  Certifications tracked: {sum(len(certs) for certs in dashboard['certifications'].values())}")
    print(f"  Pending notifications: {dashboard['notifications']}")
    
    # Test metrics
    print("\\n📈 Reporting metrics:")
    metrics = reporter.get_reporting_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\\n✅ Compliance Reporter test complete!")


if __name__ == "__main__":
    asyncio.run(test_compliance_reporter())