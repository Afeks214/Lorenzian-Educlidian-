#!/usr/bin/env python3
"""
Enterprise-Grade Audit System
Agent Zeta: Enterprise Compliance & Chaos Engineering Implementation Specialist

Advanced enterprise audit system that extends the existing blockchain audit infrastructure
with enterprise-grade features for immutable logging, forensic analysis, and compliance reporting.

Features:
- Multi-level audit trail with immutable blockchain storage
- Advanced cryptographic verification and digital signatures
- Distributed consensus for audit integrity
- Real-time compliance monitoring and alerting
- Forensic analysis capabilities for audit investigation
- Automated regulatory reporting and certification tracking
- Enterprise-grade performance and scalability
"""

import asyncio
import json
import logging
import time
import hashlib
import uuid
import ast
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import gzip
import os

# Import existing blockchain audit system
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.xai.audit.blockchain_audit import (
    BlockchainAuditSystem, AuditTransaction, AuditBlock, 
    AuditEventType, AuditLevel, AuditChain
)
from src.core.event_bus import EventBus
from src.monitoring.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SEC_REGULATION = "sec_regulation"
    GDPR = "gdpr"
    SOX = "sarbanes_oxley"
    BASEL_III = "basel_iii"
    MiFID_II = "mifid_ii"
    CFTC = "cftc"
    FINRA = "finra"
    ISO_27001 = "iso_27001"
    INTERNAL = "internal_compliance"


class AuditSeverity(Enum):
    """Enterprise audit severity levels"""
    INFORMATIONAL = "informational"
    WARNING = "warning"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class ComplianceStatus(Enum):
    """Compliance status indicators"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    INVESTIGATION_REQUIRED = "investigation_required"
    REMEDIATION_NEEDED = "remediation_needed"


@dataclass
class EnterpriseAuditEvent:
    """Enhanced audit event with enterprise features"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    
    # Core audit data
    source_system: str
    source_component: str
    decision_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Enterprise features
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    regulatory_impact: bool = False
    financial_impact: Optional[Dict[str, Any]] = None
    risk_level: int = 0  # 0-10 scale
    
    # Event payload
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Forensic metadata
    call_stack: Optional[List[str]] = None
    system_state: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    
    # Integrity and verification
    data_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    witness_signatures: List[str] = field(default_factory=list)
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if self.data_hash is None:
            self.data_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate cryptographic hash"""
        hash_data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'source_system': self.source_system,
            'source_component': self.source_component,
            'decision_id': self.decision_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'compliance_frameworks': [f.value for f in self.compliance_frameworks],
            'regulatory_impact': self.regulatory_impact,
            'financial_impact': self.financial_impact,
            'risk_level': self.risk_level,
            'event_data': self.event_data,
            'correlation_id': self.correlation_id
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode()).hexdigest()


@dataclass
class ComplianceRule:
    """Enterprise compliance rule definition"""
    rule_id: str
    name: str
    description: str
    framework: ComplianceFramework
    severity: AuditSeverity
    
    # Rule logic
    condition: str  # Python expression for rule evaluation
    action: str  # Action to take when rule is violated
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True
    last_evaluation: Optional[datetime] = None
    violation_count: int = 0


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    event_id: str
    timestamp: datetime
    severity: AuditSeverity
    
    # Violation details
    description: str
    evidence: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    
    # Status tracking
    status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


class EnterpriseAuditSystem:
    """
    Enterprise-Grade Audit System
    
    Extends blockchain audit system with enterprise features:
    - Multi-level audit trails with immutable storage
    - Advanced compliance monitoring and violation detection
    - Forensic analysis capabilities
    - Automated regulatory reporting
    - Real-time compliance dashboard
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize blockchain audit system
        self.blockchain_audit = BlockchainAuditSystem(self.config.get('blockchain_config', {}))
        
        # Enterprise components
        self.event_bus = EventBus()
        self.health_monitor = HealthMonitor()
        
        # Compliance engine
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'events_processed': 0,
            'rules_evaluated': 0,
            'violations_detected': 0,
            'avg_processing_time_ms': 0.0,
            'peak_events_per_second': 0.0,
            'compliance_score': 100.0
        }
        
        # Storage and indexing
        self.db_connection = self._initialize_database()
        self.event_index = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_enabled = True
        self.monitoring_task = None
        
        # Forensic capabilities
        self.forensic_buffer = []
        self.forensic_buffer_size = self.config.get('forensic_buffer_size', 10000)
        
        # Load compliance rules
        self._load_compliance_rules()
        
        logger.info("EnterpriseAuditSystem initialized with blockchain integration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default enterprise configuration"""
        return {
            'blockchain_config': {
                'auto_mining_enabled': True,
                'mining_interval_seconds': 30,
                'max_transactions_per_block': 50,
                'enable_digital_signatures': True,
                'enable_consensus': True
            },
            'compliance_monitoring': {
                'enabled': True,
                'real_time_evaluation': True,
                'alert_threshold': AuditSeverity.MAJOR,
                'batch_processing': False
            },
            'forensic_analysis': {
                'enabled': True,
                'buffer_size': 10000,
                'retention_days': 365,
                'compression_enabled': True
            },
            'performance': {
                'max_events_per_second': 1000,
                'processing_timeout_ms': 5000,
                'batch_size': 100
            },
            'storage': {
                'database_path': '/tmp/enterprise_audit.db',
                'backup_enabled': True,
                'backup_interval_hours': 6
            }
        }
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize enterprise audit database"""
        db_path = self.config['storage']['database_path']
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create enterprise audit tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS enterprise_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                source_system TEXT NOT NULL,
                source_component TEXT NOT NULL,
                decision_id TEXT,
                user_id TEXT,
                session_id TEXT,
                compliance_frameworks TEXT,
                regulatory_impact INTEGER,
                financial_impact TEXT,
                risk_level INTEGER,
                event_data TEXT,
                correlation_id TEXT,
                data_hash TEXT,
                digital_signature TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_violations (
                violation_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                evidence TEXT,
                impact_assessment TEXT,
                status TEXT,
                assigned_to TEXT,
                resolution_notes TEXT,
                resolved_at TEXT,
                FOREIGN KEY (event_id) REFERENCES enterprise_events (event_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                framework TEXT NOT NULL,
                severity TEXT NOT NULL,
                condition TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT,
                active INTEGER,
                last_evaluation TEXT,
                violation_count INTEGER
            )
        ''')
        
        # Create indices for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON enterprise_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON enterprise_events(event_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_severity ON enterprise_events(severity)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_violations_status ON compliance_violations(status)')
        
        conn.commit()
        return conn
    
    def _load_compliance_rules(self):
        """Load compliance rules from configuration"""
        default_rules = [
            ComplianceRule(
                rule_id="SEC_001",
                name="Trading Decision Documentation",
                description="All trading decisions must be documented with rationale",
                framework=ComplianceFramework.SEC_REGULATION,
                severity=AuditSeverity.MAJOR,
                condition="event_type == 'decision_made' and not event_data.get('rationale')",
                action="create_violation"
            ),
            ComplianceRule(
                rule_id="GDPR_001",
                name="Personal Data Access Logging",
                description="All personal data access must be logged",
                framework=ComplianceFramework.GDPR,
                severity=AuditSeverity.MAJOR,
                condition="'personal_data' in event_data and not event_data.get('access_logged')",
                action="create_violation"
            ),
            ComplianceRule(
                rule_id="SOX_001",
                name="Financial Control Verification",
                description="Financial controls must be verified and documented",
                framework=ComplianceFramework.SOX,
                severity=AuditSeverity.CRITICAL,
                condition="financial_impact and not event_data.get('control_verified')",
                action="create_violation"
            ),
            ComplianceRule(
                rule_id="INTERNAL_001",
                name="High-Risk Decision Approval",
                description="High-risk decisions require approval",
                framework=ComplianceFramework.INTERNAL,
                severity=AuditSeverity.MAJOR,
                condition="risk_level > 7 and not event_data.get('approved_by')",
                action="create_violation"
            )
        ]
        
        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    async def log_enterprise_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        source_system: str,
        source_component: str,
        event_data: Dict[str, Any],
        decision_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None,
        regulatory_impact: bool = False,
        financial_impact: Optional[Dict[str, Any]] = None,
        risk_level: int = 0,
        correlation_id: Optional[str] = None
    ) -> EnterpriseAuditEvent:
        """
        Log enterprise audit event with enhanced metadata
        
        Args:
            event_type: Type of audit event
            severity: Enterprise severity level
            source_system: Source system identifier
            source_component: Source component identifier
            event_data: Event payload data
            decision_id: Associated decision ID
            user_id: User identifier
            session_id: Session identifier
            compliance_frameworks: Applicable compliance frameworks
            regulatory_impact: Whether event has regulatory impact
            financial_impact: Financial impact assessment
            risk_level: Risk level (0-10)
            correlation_id: Correlation identifier for tracking
            
        Returns:
            EnterpriseAuditEvent: Created event
        """
        start_time = time.time()
        
        # Create enterprise event
        event = EnterpriseAuditEvent(
            event_id=f"ent_{uuid.uuid4().hex}",
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            source_system=source_system,
            source_component=source_component,
            decision_id=decision_id,
            user_id=user_id,
            session_id=session_id,
            compliance_frameworks=compliance_frameworks or [],
            regulatory_impact=regulatory_impact,
            financial_impact=financial_impact,
            risk_level=risk_level,
            event_data=event_data,
            correlation_id=correlation_id
        )
        
        # Store in database
        await self._store_event(event)
        
        # Add to blockchain
        await self._add_to_blockchain(event)
        
        # Add to forensic buffer
        self.forensic_buffer.append(event)
        if len(self.forensic_buffer) > self.forensic_buffer_size:
            self.forensic_buffer.pop(0)
        
        # Evaluate compliance rules
        if self.config['compliance_monitoring']['enabled']:
            await self._evaluate_compliance_rules(event)
        
        # Update performance metrics
        processing_time = (time.time() - start_time) * 1000
        self.performance_metrics['events_processed'] += 1
        
        # Update average processing time
        total_events = self.performance_metrics['events_processed']
        old_avg = self.performance_metrics['avg_processing_time_ms']
        self.performance_metrics['avg_processing_time_ms'] = (
            (old_avg * (total_events - 1) + processing_time) / total_events
        )
        
        # Emit event
        await self.event_bus.emit("enterprise_audit_event", {
            "event": event,
            "processing_time_ms": processing_time
        })
        
        logger.info(f"Enterprise audit event logged: {event.event_id}")
        return event
    
    async def _store_event(self, event: EnterpriseAuditEvent):
        """Store event in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO enterprise_events (
                event_id, timestamp, event_type, severity, source_system, 
                source_component, decision_id, user_id, session_id, 
                compliance_frameworks, regulatory_impact, financial_impact,
                risk_level, event_data, correlation_id, data_hash, 
                digital_signature, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type.value,
            event.severity.value,
            event.source_system,
            event.source_component,
            event.decision_id,
            event.user_id,
            event.session_id,
            json.dumps([f.value for f in event.compliance_frameworks]),
            int(event.regulatory_impact),
            json.dumps(event.financial_impact) if event.financial_impact else None,
            event.risk_level,
            json.dumps(event.event_data),
            event.correlation_id,
            event.data_hash,
            event.digital_signature,
            event.created_at.isoformat(),
            event.updated_at.isoformat()
        ))
        self.db_connection.commit()
    
    async def _add_to_blockchain(self, event: EnterpriseAuditEvent):
        """Add event to blockchain audit trail"""
        # Convert to blockchain audit level
        audit_level_map = {
            AuditSeverity.INFORMATIONAL: AuditLevel.LOW,
            AuditSeverity.WARNING: AuditLevel.LOW,
            AuditSeverity.MINOR: AuditLevel.MEDIUM,
            AuditSeverity.MAJOR: AuditLevel.HIGH,
            AuditSeverity.CRITICAL: AuditLevel.HIGH,
            AuditSeverity.CATASTROPHIC: AuditLevel.HIGH
        }
        
        await self.blockchain_audit.log_audit_event(
            event_type=event.event_type,
            audit_level=audit_level_map.get(event.severity, AuditLevel.MEDIUM),
            source_system=event.source_system,
            source_component=event.source_component,
            event_data={
                "enterprise_event_id": event.event_id,
                "severity": event.severity.value,
                "compliance_frameworks": [f.value for f in event.compliance_frameworks],
                "regulatory_impact": event.regulatory_impact,
                "financial_impact": event.financial_impact,
                "risk_level": event.risk_level,
                "event_data": event.event_data,
                "correlation_id": event.correlation_id
            },
            decision_id=event.decision_id,
            user_id=event.user_id,
            session_id=event.session_id
        )
    
    async def _evaluate_compliance_rules(self, event: EnterpriseAuditEvent):
        """Evaluate compliance rules against event"""
        for rule_id, rule in self.compliance_rules.items():
            if not rule.active:
                continue
            
            try:
                # Create evaluation context
                eval_context = {
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'source_system': event.source_system,
                    'source_component': event.source_component,
                    'event_data': event.event_data,
                    'compliance_frameworks': [f.value for f in event.compliance_frameworks],
                    'regulatory_impact': event.regulatory_impact,
                    'financial_impact': event.financial_impact,
                    'risk_level': event.risk_level,
                    'user_id': event.user_id
                }
                
                # Evaluate rule condition - using safer AST-based evaluator instead of eval()
                if self._safe_eval_condition(rule.condition, eval_context):
                    # Rule violated
                    violation = ComplianceViolation(
                        violation_id=f"viol_{uuid.uuid4().hex}",
                        rule_id=rule_id,
                        event_id=event.event_id,
                        timestamp=datetime.now(timezone.utc),
                        severity=rule.severity,
                        description=f"Compliance rule violated: {rule.name}",
                        evidence=eval_context,
                        impact_assessment=await self._assess_violation_impact(rule, event)
                    )
                    
                    # Store violation
                    await self._store_violation(violation)
                    self.violations[violation.violation_id] = violation
                    
                    # Update rule statistics
                    rule.violation_count += 1
                    rule.last_evaluation = datetime.now(timezone.utc)
                    
                    # Update performance metrics
                    self.performance_metrics['violations_detected'] += 1
                    
                    # Emit violation event
                    await self.event_bus.emit("compliance_violation", {
                        "violation": violation,
                        "rule": rule,
                        "event": event
                    })
                    
                    logger.warning(f"Compliance violation detected: {violation.violation_id}")
                
                self.performance_metrics['rules_evaluated'] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating compliance rule {rule_id}: {e}")
    
    def _safe_eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a condition expression using AST parsing instead of eval().
        
        Args:
            condition: The condition string to evaluate
            context: Variable context for evaluation
            
        Returns:
            bool: Result of condition evaluation
        """
        try:
            # Parse the condition into an AST
            tree = ast.parse(condition, mode='eval')
            
            # Evaluate the AST safely
            return self._eval_ast_node(tree.body, context)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _eval_ast_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate an AST node.
        
        Args:
            node: AST node to evaluate
            context: Variable context
            
        Returns:
            Evaluation result
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return context.get(node.id, None)
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left, context)
            right = self._eval_ast_node(node.right, context)
            
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            elif isinstance(node.op, ast.LShift):
                return left << right
            elif isinstance(node.op, ast.RShift):
                return left >> right
            elif isinstance(node.op, ast.BitOr):
                return left | right
            elif isinstance(node.op, ast.BitXor):
                return left ^ right
            elif isinstance(node.op, ast.BitAnd):
                return left & right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
        elif isinstance(node, ast.Compare):
            left = self._eval_ast_node(node.left, context)
            result = True
            
            for comparator, op in zip(node.comparators, node.ops):
                right = self._eval_ast_node(comparator, context)
                
                if isinstance(op, ast.Eq):
                    result = result and (left == right)
                elif isinstance(op, ast.NotEq):
                    result = result and (left != right)
                elif isinstance(op, ast.Lt):
                    result = result and (left < right)
                elif isinstance(op, ast.LtE):
                    result = result and (left <= right)
                elif isinstance(op, ast.Gt):
                    result = result and (left > right)
                elif isinstance(op, ast.GtE):
                    result = result and (left >= right)
                elif isinstance(op, ast.Is):
                    result = result and (left is right)
                elif isinstance(op, ast.IsNot):
                    result = result and (left is not right)
                elif isinstance(op, ast.In):
                    result = result and (left in right)
                elif isinstance(op, ast.NotIn):
                    result = result and (left not in right)
                
                if not result:
                    break
                left = right
            
            return result
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval_ast_node(value, context) for value in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self._eval_ast_node(value, context) for value in node.values)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand, context)
            
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.Not):
                return not operand
            elif isinstance(node.op, ast.Invert):
                return ~operand
        elif isinstance(node, ast.List):
            return [self._eval_ast_node(item, context) for item in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_ast_node(item, context) for item in node.elts)
        elif isinstance(node, ast.Dict):
            return {
                self._eval_ast_node(k, context): self._eval_ast_node(v, context)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Subscript):
            value = self._eval_ast_node(node.value, context)
            slice_value = self._eval_ast_node(node.slice, context)
            return value[slice_value]
        elif isinstance(node, ast.Attribute):
            value = self._eval_ast_node(node.value, context)
            return getattr(value, node.attr)
        
        # If we get here, the node type is not supported
        raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    async def _assess_violation_impact(self, rule: ComplianceRule, event: EnterpriseAuditEvent) -> Dict[str, Any]:
        """Assess impact of compliance violation"""
        impact = {
            'regulatory_risk': 'HIGH' if event.regulatory_impact else 'LOW',
            'financial_risk': 'HIGH' if event.financial_impact else 'LOW',
            'operational_risk': 'MEDIUM' if event.risk_level > 5 else 'LOW',
            'reputational_risk': 'HIGH' if rule.severity in [AuditSeverity.CRITICAL, AuditSeverity.CATASTROPHIC] else 'MEDIUM'
        }
        
        # Calculate overall risk score
        risk_weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        total_risk = sum(risk_weights.get(risk, 0) for risk in impact.values())
        max_risk = len(impact) * 3
        
        impact['overall_risk_score'] = total_risk / max_risk
        impact['recommended_action'] = self._recommend_action(impact['overall_risk_score'])
        
        return impact
    
    def _recommend_action(self, risk_score: float) -> str:
        """Recommend action based on risk score"""
        if risk_score >= 0.8:
            return "IMMEDIATE_ESCALATION"
        elif risk_score >= 0.6:
            return "URGENT_REVIEW"
        elif risk_score >= 0.4:
            return "STANDARD_REVIEW"
        else:
            return "MONITOR"
    
    async def _store_violation(self, violation: ComplianceViolation):
        """Store violation in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO compliance_violations (
                violation_id, rule_id, event_id, timestamp, severity,
                description, evidence, impact_assessment, status,
                assigned_to, resolution_notes, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            violation.violation_id,
            violation.rule_id,
            violation.event_id,
            violation.timestamp.isoformat(),
            violation.severity.value,
            violation.description,
            json.dumps(violation.evidence),
            json.dumps(violation.impact_assessment),
            violation.status.value,
            violation.assigned_to,
            violation.resolution_notes,
            violation.resolved_at.isoformat() if violation.resolved_at else None
        ))
        self.db_connection.commit()
    
    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get real-time compliance dashboard data"""
        cursor = self.db_connection.cursor()
        
        # Get violation statistics
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM compliance_violations
            GROUP BY status
        ''')
        violation_stats = dict(cursor.fetchall())
        
        # Get event statistics by severity
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM enterprise_events
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY severity
        ''')
        event_stats = dict(cursor.fetchall())
        
        # Get top violation rules
        cursor.execute('''
            SELECT rule_id, COUNT(*) as count
            FROM compliance_violations
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY rule_id
            ORDER BY count DESC
            LIMIT 10
        ''')
        top_violations = cursor.fetchall()
        
        # Calculate compliance score
        total_events = sum(event_stats.values())
        total_violations = sum(violation_stats.values())
        compliance_score = 100.0 * (1 - (total_violations / max(total_events, 1)))
        
        return {
            'compliance_score': compliance_score,
            'violation_statistics': violation_stats,
            'event_statistics': event_stats,
            'top_violations': top_violations,
            'performance_metrics': self.performance_metrics.copy(),
            'blockchain_status': self.blockchain_audit.get_system_status(),
            'total_events_24h': total_events,
            'total_violations_7d': total_violations
        }
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        include_remediation: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        cursor = self.db_connection.cursor()
        
        # Get events for framework
        cursor.execute('''
            SELECT * FROM enterprise_events
            WHERE compliance_frameworks LIKE ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (f'%{framework.value}%', start_date.isoformat(), end_date.isoformat()))
        
        events = cursor.fetchall()
        
        # Get violations for framework
        cursor.execute('''
            SELECT v.*, r.framework FROM compliance_violations v
            JOIN compliance_rules r ON v.rule_id = r.rule_id
            WHERE r.framework = ?
            AND v.timestamp BETWEEN ? AND ?
            ORDER BY v.timestamp DESC
        ''', (framework.value, start_date.isoformat(), end_date.isoformat()))
        
        violations = cursor.fetchall()
        
        # Generate report
        report = {
            'report_id': f"compliance_{framework.value}_{uuid.uuid4().hex[:8]}",
            'framework': framework.value,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_events': len(events),
                'total_violations': len(violations),
                'compliance_score': 100.0 * (1 - (len(violations) / max(len(events), 1))),
                'high_risk_violations': len([v for v in violations if v[4] in ['critical', 'catastrophic']]),
                'resolved_violations': len([v for v in violations if v[8] == 'resolved'])
            },
            'events': events,
            'violations': violations,
            'blockchain_integrity': self.blockchain_audit.verify_audit_integrity()
        }
        
        if include_remediation:
            report['remediation_recommendations'] = await self._generate_remediation_recommendations(violations)
        
        return report
    
    async def _generate_remediation_recommendations(self, violations: List[Any]) -> List[Dict[str, Any]]:
        """Generate remediation recommendations"""
        recommendations = []
        
        # Group violations by rule
        rule_violations = defaultdict(list)
        for violation in violations:
            rule_violations[violation[1]].append(violation)
        
        for rule_id, rule_violations_list in rule_violations.items():
            if len(rule_violations_list) > 1:
                recommendations.append({
                    'type': 'PROCESS_IMPROVEMENT',
                    'priority': 'HIGH',
                    'rule_id': rule_id,
                    'description': f"Multiple violations detected for rule {rule_id}. Consider process improvements.",
                    'violation_count': len(rule_violations_list),
                    'recommended_actions': [
                        'Review and update process documentation',
                        'Implement additional controls',
                        'Provide additional training',
                        'Consider automation opportunities'
                    ]
                })
        
        return recommendations
    
    async def forensic_search(
        self,
        query: Dict[str, Any],
        timeframe: Optional[Tuple[datetime, datetime]] = None,
        correlation_id: Optional[str] = None
    ) -> List[EnterpriseAuditEvent]:
        """Advanced forensic search capabilities"""
        cursor = self.db_connection.cursor()
        
        # Build query
        where_conditions = []
        params = []
        
        if timeframe:
            where_conditions.append("timestamp BETWEEN ? AND ?")
            params.extend([timeframe[0].isoformat(), timeframe[1].isoformat()])
        
        if correlation_id:
            where_conditions.append("correlation_id = ?")
            params.append(correlation_id)
        
        # Add query conditions
        for key, value in query.items():
            if key == 'event_type':
                where_conditions.append("event_type = ?")
                params.append(value)
            elif key == 'severity':
                where_conditions.append("severity = ?")
                params.append(value)
            elif key == 'source_system':
                where_conditions.append("source_system = ?")
                params.append(value)
            elif key == 'user_id':
                where_conditions.append("user_id = ?")
                params.append(value)
            elif key == 'risk_level_min':
                where_conditions.append("risk_level >= ?")
                params.append(value)
        
        # Execute query
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        cursor.execute(f'''
            SELECT * FROM enterprise_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT 1000
        ''', params)
        
        results = cursor.fetchall()
        
        # Convert to events (simplified conversion)
        events = []
        for row in results:
            # Create basic event object for forensic analysis
            event_data = {
                'event_id': row[0],
                'timestamp': row[1],
                'event_type': row[2],
                'severity': row[3],
                'source_system': row[4],
                'source_component': row[5],
                'correlation_id': row[14]
            }
            events.append(event_data)
        
        return events
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Enterprise audit monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        logger.info("Enterprise audit monitoring stopped")
    
    async def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Check system health
                await self._check_system_health()
                
                # Process any pending compliance evaluations
                await self._process_pending_evaluations()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(1)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_system_health(self):
        """Check system health indicators"""
        # Check database connection
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Check blockchain health
        blockchain_status = self.blockchain_audit.get_system_status()
        if not blockchain_status.get('audit_chain', {}).get('chain_integrity', False):
            logger.error("Blockchain integrity check failed")
    
    async def _process_pending_evaluations(self):
        """Process any pending compliance evaluations"""
        # This would process any backlog of compliance evaluations
        pass
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        # Update events per second
        current_time = time.time()
        if hasattr(self, '_last_metric_update'):
            time_diff = current_time - self._last_metric_update
            if time_diff > 0:
                events_since_last = self.performance_metrics['events_processed'] - getattr(self, '_last_event_count', 0)
                events_per_second = events_since_last / time_diff
                self.performance_metrics['peak_events_per_second'] = max(
                    self.performance_metrics['peak_events_per_second'],
                    events_per_second
                )
        
        self._last_metric_update = current_time
        self._last_event_count = self.performance_metrics['events_processed']
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()


# Test function
async def test_enterprise_audit_system():
    """Test the Enterprise Audit System"""
    print("🧪 Testing Enterprise Audit System")
    
    # Initialize system
    audit_system = EnterpriseAuditSystem()
    
    # Start monitoring
    await audit_system.start_monitoring()
    
    try:
        # Test enterprise event logging
        print("\\n📝 Testing enterprise event logging...")
        
        # Log high-risk trading decision
        event = await audit_system.log_enterprise_event(
            event_type=AuditEventType.DECISION_MADE,
            severity=AuditSeverity.MAJOR,
            source_system="STRATEGIC_MARL",
            source_component="DecisionMaker",
            event_data={
                "action": "LONG",
                "confidence": 0.95,
                "position_size": 1000000,
                "rationale": "Strong momentum signals with confirmed breakout"
            },
            decision_id="dec_enterprise_001",
            user_id="trader_001",
            compliance_frameworks=[ComplianceFramework.SEC_REGULATION, ComplianceFramework.FINRA],
            regulatory_impact=True,
            financial_impact={"estimated_pnl": 50000, "max_loss": 25000},
            risk_level=8,
            correlation_id="corr_001"
        )
        
        print(f"Logged enterprise event: {event.event_id}")
        
        # Log event that will trigger compliance violation
        violation_event = await audit_system.log_enterprise_event(
            event_type=AuditEventType.DECISION_MADE,
            severity=AuditSeverity.CRITICAL,
            source_system="TACTICAL_MARL",
            source_component="DecisionMaker",
            event_data={
                "action": "SHORT",
                "confidence": 0.9,
                "position_size": 2000000
                # Missing rationale - will trigger SEC_001 violation
            },
            decision_id="dec_enterprise_002",
            user_id="trader_002",
            compliance_frameworks=[ComplianceFramework.SEC_REGULATION],
            regulatory_impact=True,
            financial_impact={"estimated_pnl": 75000, "max_loss": 100000},
            risk_level=9,
            correlation_id="corr_002"
        )
        
        print(f"Logged violation event: {violation_event.event_id}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Test compliance dashboard
        print("\\n📊 Testing compliance dashboard...")
        dashboard = await audit_system.get_compliance_dashboard()
        print(f"Compliance score: {dashboard['compliance_score']:.2f}%")
        print(f"Total violations: {sum(dashboard['violation_statistics'].values())}")
        print(f"Performance metrics: {dashboard['performance_metrics']}")
        
        # Test forensic search
        print("\\n🔍 Testing forensic search...")
        search_results = await audit_system.forensic_search(
            query={'event_type': 'decision_made', 'risk_level_min': 8},
            timeframe=(datetime.now(timezone.utc) - timedelta(hours=1), datetime.now(timezone.utc))
        )
        print(f"Forensic search found {len(search_results)} high-risk events")
        
        # Test compliance report generation
        print("\\n📋 Testing compliance report generation...")
        report = await audit_system.generate_compliance_report(
            framework=ComplianceFramework.SEC_REGULATION,
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc)
        )
        print(f"Generated compliance report: {report['report_id']}")
        print(f"Report summary: {report['summary']}")
        
        print("\\n✅ Enterprise Audit System test complete!")
        
    finally:
        await audit_system.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(test_enterprise_audit_system())