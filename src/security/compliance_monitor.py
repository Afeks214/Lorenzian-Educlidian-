"""
Comprehensive Compliance Monitoring System
Real-time monitoring and validation for financial regulations
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
from decimal import Decimal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from src.monitoring.logger_config import get_logger
from src.security.financial_audit_logger import (
    log_audit_event, AuditEventType, AuditSeverity, AuditContext,
    ComplianceFramework, FinancialData
)
from src.security.rbac_system import rbac_system, PermissionScope
from src.security.vault_integration import get_vault_secret

logger = get_logger(__name__)

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class RuleType(Enum):
    """Types of compliance rules"""
    POSITION_LIMIT = "position_limit"
    RISK_LIMIT = "risk_limit"
    TRADING_WINDOW = "trading_window"
    VOLUME_LIMIT = "volume_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    DOCUMENTATION = "documentation"
    SEGREGATION = "segregation"
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"
    BEST_EXECUTION = "best_execution"
    LIQUIDITY = "liquidity"
    CAPITAL_ADEQUACY = "capital_adequacy"
    STRESS_TESTING = "stress_testing"
    REPORTING = "reporting"
    DATA_PROTECTION = "data_protection"
    RETENTION = "retention"

class MonitoringFrequency(Enum):
    """Monitoring frequency options"""
    REAL_TIME = "real_time"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    compliance_framework: ComplianceFramework
    monitoring_frequency: MonitoringFrequency
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "compliance_framework": self.compliance_framework.value,
            "monitoring_frequency": self.monitoring_frequency.value,
            "parameters": self.parameters,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    rule_name: str
    compliance_framework: ComplianceFramework
    severity: ComplianceStatus
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detected_by: str = "compliance_monitor"
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "compliance_framework": self.compliance_framework.value,
            "severity": self.severity.value,
            "description": self.description,
            "details": self.details,
            "occurred_at": self.occurred_at.isoformat(),
            "detected_by": self.detected_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes
        }

@dataclass
class ComplianceMetric:
    """Compliance metric tracking"""
    metric_id: str
    rule_id: str
    name: str
    value: Union[float, int, str]
    threshold: Union[float, int, str]
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ComplianceStatus = ComplianceStatus.COMPLIANT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "rule_id": self.rule_id,
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value
        }

class ComplianceMonitor:
    """Comprehensive compliance monitoring system"""
    
    def __init__(self, storage_path: str = "compliance_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Rule storage
        self.rules: Dict[str, ComplianceRule] = {}
        self.violations: List[ComplianceViolation] = []
        self.metrics: List[ComplianceMetric] = []
        
        # Monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Rule evaluators
        self.rule_evaluators: Dict[RuleType, Callable] = {
            RuleType.POSITION_LIMIT: self._evaluate_position_limit,
            RuleType.RISK_LIMIT: self._evaluate_risk_limit,
            RuleType.TRADING_WINDOW: self._evaluate_trading_window,
            RuleType.VOLUME_LIMIT: self._evaluate_volume_limit,
            RuleType.CONCENTRATION_LIMIT: self._evaluate_concentration_limit,
            RuleType.DOCUMENTATION: self._evaluate_documentation,
            RuleType.SEGREGATION: self._evaluate_segregation,
            RuleType.MARKET_MANIPULATION: self._evaluate_market_manipulation,
            RuleType.INSIDER_TRADING: self._evaluate_insider_trading,
            RuleType.BEST_EXECUTION: self._evaluate_best_execution,
            RuleType.LIQUIDITY: self._evaluate_liquidity,
            RuleType.CAPITAL_ADEQUACY: self._evaluate_capital_adequacy,
            RuleType.STRESS_TESTING: self._evaluate_stress_testing,
            RuleType.REPORTING: self._evaluate_reporting,
            RuleType.DATA_PROTECTION: self._evaluate_data_protection,
            RuleType.RETENTION: self._evaluate_retention
        }
        
        # Thread pool for intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load existing data
        self._load_data()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("ComplianceMonitor initialized")
    
    def _load_data(self):
        """Load compliance data from storage"""
        # Load rules
        rules_file = self.storage_path / "rules.json"
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                    for rule_data in rules_data:
                        rule = ComplianceRule(
                            rule_id=rule_data["rule_id"],
                            name=rule_data["name"],
                            description=rule_data["description"],
                            rule_type=RuleType(rule_data["rule_type"]),
                            compliance_framework=ComplianceFramework(rule_data["compliance_framework"]),
                            monitoring_frequency=MonitoringFrequency(rule_data["monitoring_frequency"]),
                            parameters=rule_data["parameters"],
                            is_active=rule_data["is_active"],
                            created_at=datetime.fromisoformat(rule_data["created_at"]),
                            updated_at=datetime.fromisoformat(rule_data["updated_at"]),
                            created_by=rule_data["created_by"]
                        )
                        self.rules[rule.rule_id] = rule
            except Exception as e:
                logger.error(f"Failed to load compliance rules: {e}")
        
        # Load violations
        violations_file = self.storage_path / "violations.json"
        if violations_file.exists():
            try:
                with open(violations_file, 'r') as f:
                    violations_data = json.load(f)
                    for violation_data in violations_data:
                        violation = ComplianceViolation(
                            violation_id=violation_data["violation_id"],
                            rule_id=violation_data["rule_id"],
                            rule_name=violation_data["rule_name"],
                            compliance_framework=ComplianceFramework(violation_data["compliance_framework"]),
                            severity=ComplianceStatus(violation_data["severity"]),
                            description=violation_data["description"],
                            details=violation_data["details"],
                            occurred_at=datetime.fromisoformat(violation_data["occurred_at"]),
                            detected_by=violation_data["detected_by"],
                            resolved_at=datetime.fromisoformat(violation_data["resolved_at"]) if violation_data["resolved_at"] else None,
                            resolved_by=violation_data["resolved_by"],
                            resolution_notes=violation_data["resolution_notes"]
                        )
                        self.violations.append(violation)
            except Exception as e:
                logger.error(f"Failed to load compliance violations: {e}")
    
    def _save_data(self):
        """Save compliance data to storage"""
        # Save rules
        rules_file = self.storage_path / "rules.json"
        try:
            with open(rules_file, 'w') as f:
                rules_data = [rule.to_dict() for rule in self.rules.values()]
                json.dump(rules_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save compliance rules: {e}")
        
        # Save violations
        violations_file = self.storage_path / "violations.json"
        try:
            with open(violations_file, 'w') as f:
                violations_data = [violation.to_dict() for violation in self.violations]
                json.dump(violations_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save compliance violations: {e}")
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        default_rules = [
            # SOX Rules
            ComplianceRule(
                rule_id="sox_internal_controls",
                name="SOX Internal Controls",
                description="Verify internal controls for financial reporting",
                rule_type=RuleType.DOCUMENTATION,
                compliance_framework=ComplianceFramework.SOX,
                monitoring_frequency=MonitoringFrequency.QUARTERLY,
                parameters={"required_controls": ["segregation_of_duties", "authorization_limits", "documentation"]}
            ),
            ComplianceRule(
                rule_id="sox_ceo_cfo_certification",
                name="CEO/CFO Certification",
                description="Ensure CEO/CFO certification of financial statements",
                rule_type=RuleType.DOCUMENTATION,
                compliance_framework=ComplianceFramework.SOX,
                monitoring_frequency=MonitoringFrequency.QUARTERLY,
                parameters={"required_certifications": ["accuracy", "completeness", "internal_controls"]}
            ),
            
            # SEC Rules
            ComplianceRule(
                rule_id="sec_position_limits",
                name="SEC Position Limits",
                description="Monitor position limits per SEC regulations",
                rule_type=RuleType.POSITION_LIMIT,
                compliance_framework=ComplianceFramework.SEC,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"max_position_percent": 10.0, "reporting_threshold": 5.0}
            ),
            ComplianceRule(
                rule_id="sec_insider_trading",
                name="SEC Insider Trading",
                description="Monitor for insider trading violations",
                rule_type=RuleType.INSIDER_TRADING,
                compliance_framework=ComplianceFramework.SEC,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"blackout_periods": [], "restricted_persons": []}
            ),
            
            # CFTC Rules
            ComplianceRule(
                rule_id="cftc_position_limits",
                name="CFTC Position Limits",
                description="Monitor position limits for derivatives",
                rule_type=RuleType.POSITION_LIMIT,
                compliance_framework=ComplianceFramework.CFTC,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"max_position_contracts": 10000, "reporting_threshold": 5000}
            ),
            ComplianceRule(
                rule_id="cftc_risk_management",
                name="CFTC Risk Management",
                description="Ensure adequate risk management procedures",
                rule_type=RuleType.RISK_LIMIT,
                compliance_framework=ComplianceFramework.CFTC,
                monitoring_frequency=MonitoringFrequency.DAILY,
                parameters={"max_var": 1000000, "stress_test_frequency": "weekly"}
            ),
            
            # FINRA Rules
            ComplianceRule(
                rule_id="finra_best_execution",
                name="FINRA Best Execution",
                description="Ensure best execution of customer orders",
                rule_type=RuleType.BEST_EXECUTION,
                compliance_framework=ComplianceFramework.FINRA,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"execution_quality_threshold": 0.95}
            ),
            ComplianceRule(
                rule_id="finra_supervisory_procedures",
                name="FINRA Supervisory Procedures",
                description="Maintain supervisory procedures",
                rule_type=RuleType.DOCUMENTATION,
                compliance_framework=ComplianceFramework.FINRA,
                monitoring_frequency=MonitoringFrequency.ANNUALLY,
                parameters={"required_procedures": ["trade_review", "exception_reporting", "compliance_testing"]}
            ),
            
            # MiFID2 Rules
            ComplianceRule(
                rule_id="mifid2_transaction_reporting",
                name="MiFID2 Transaction Reporting",
                description="Report transactions to regulatory authorities",
                rule_type=RuleType.REPORTING,
                compliance_framework=ComplianceFramework.MiFID2,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"reporting_deadline_minutes": 15}
            ),
            ComplianceRule(
                rule_id="mifid2_best_execution",
                name="MiFID2 Best Execution",
                description="Demonstrate best execution for client orders",
                rule_type=RuleType.BEST_EXECUTION,
                compliance_framework=ComplianceFramework.MiFID2,
                monitoring_frequency=MonitoringFrequency.REAL_TIME,
                parameters={"execution_factors": ["price", "speed", "likelihood", "size", "nature"]}
            ),
            
            # GDPR Rules
            ComplianceRule(
                rule_id="gdpr_data_protection",
                name="GDPR Data Protection",
                description="Ensure personal data protection",
                rule_type=RuleType.DATA_PROTECTION,
                compliance_framework=ComplianceFramework.GDPR,
                monitoring_frequency=MonitoringFrequency.DAILY,
                parameters={"encryption_required": True, "access_logging": True}
            ),
            ComplianceRule(
                rule_id="gdpr_data_retention",
                name="GDPR Data Retention",
                description="Comply with data retention limits",
                rule_type=RuleType.RETENTION,
                compliance_framework=ComplianceFramework.GDPR,
                monitoring_frequency=MonitoringFrequency.MONTHLY,
                parameters={"max_retention_days": 2555, "deletion_required": True}  # 7 years
            )
        ]
        
        for rule in default_rules:
            if rule.rule_id not in self.rules:
                self.rules[rule.rule_id] = rule
        
        self._save_data()
    
    async def start_monitoring(self):
        """Start compliance monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring tasks for each active rule
        for rule_id, rule in self.rules.items():
            if rule.is_active:
                self.monitoring_tasks[rule_id] = asyncio.create_task(
                    self._monitor_rule(rule)
                )
        
        # Start cleanup task
        self.monitoring_tasks["cleanup"] = asyncio.create_task(self._cleanup_task())
        
        logger.info(f"Compliance monitoring started for {len(self.monitoring_tasks)} rules")
    
    async def stop_monitoring(self):
        """Stop compliance monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        logger.info("Compliance monitoring stopped")
    
    async def _monitor_rule(self, rule: ComplianceRule):
        """Monitor a specific compliance rule"""
        while self.running:
            try:
                # Get monitoring interval
                interval = self._get_monitoring_interval(rule.monitoring_frequency)
                
                # Evaluate rule
                await self._evaluate_rule(rule)
                
                # Wait for next evaluation
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring rule {rule.rule_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _get_monitoring_interval(self, frequency: MonitoringFrequency) -> int:
        """Get monitoring interval in seconds"""
        intervals = {
            MonitoringFrequency.REAL_TIME: 1,
            MonitoringFrequency.MINUTELY: 60,
            MonitoringFrequency.HOURLY: 3600,
            MonitoringFrequency.DAILY: 86400,
            MonitoringFrequency.WEEKLY: 604800,
            MonitoringFrequency.MONTHLY: 2592000,
            MonitoringFrequency.QUARTERLY: 7776000,
            MonitoringFrequency.ANNUALLY: 31536000
        }
        return intervals.get(frequency, 3600)
    
    async def _evaluate_rule(self, rule: ComplianceRule):
        """Evaluate a compliance rule"""
        evaluator = self.rule_evaluators.get(rule.rule_type)
        if not evaluator:
            logger.warning(f"No evaluator for rule type: {rule.rule_type}")
            return
        
        try:
            # Run evaluation
            result = await evaluator(rule)
            
            # Record metric
            if result:
                metric = ComplianceMetric(
                    metric_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    name=rule.name,
                    value=result.get("value", 0),
                    threshold=result.get("threshold", 0),
                    unit=result.get("unit", ""),
                    status=result.get("status", ComplianceStatus.COMPLIANT)
                )
                
                self.metrics.append(metric)
                
                # Check for violations
                if metric.status in [ComplianceStatus.VIOLATION, ComplianceStatus.CRITICAL]:
                    await self._record_violation(rule, result)
                    
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _record_violation(self, rule: ComplianceRule, result: Dict[str, Any]):
        """Record a compliance violation"""
        violation = ComplianceViolation(
            violation_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            compliance_framework=rule.compliance_framework,
            severity=result.get("status", ComplianceStatus.VIOLATION),
            description=result.get("description", f"Violation of {rule.name}"),
            details=result.get("details", {})
        )
        
        self.violations.append(violation)
        self._save_data()
        
        # Audit log
        await log_audit_event(
            AuditEventType.COMPLIANCE_VIOLATION,
            f"Compliance violation: {rule.name}",
            severity=AuditSeverity.HIGH,
            additional_data={
                "rule_id": rule.rule_id,
                "compliance_framework": rule.compliance_framework.value,
                "violation_details": violation.details
            }
        )
        
        logger.warning(f"Compliance violation recorded: {violation.violation_id}")
    
    # Rule evaluators
    async def _evaluate_position_limit(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate position limit rule"""
        # This would integrate with actual position data
        # For now, return a mock evaluation
        max_position = rule.parameters.get("max_position_percent", 10.0)
        current_position = 8.5  # Mock current position
        
        status = ComplianceStatus.COMPLIANT
        if current_position > max_position:
            status = ComplianceStatus.VIOLATION
        elif current_position > max_position * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_position,
            "threshold": max_position,
            "unit": "percent",
            "status": status,
            "description": f"Current position: {current_position}%, Limit: {max_position}%"
        }
    
    async def _evaluate_risk_limit(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate risk limit rule"""
        max_var = rule.parameters.get("max_var", 1000000)
        current_var = 850000  # Mock current VaR
        
        status = ComplianceStatus.COMPLIANT
        if current_var > max_var:
            status = ComplianceStatus.VIOLATION
        elif current_var > max_var * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_var,
            "threshold": max_var,
            "unit": "USD",
            "status": status,
            "description": f"Current VaR: ${current_var:,.2f}, Limit: ${max_var:,.2f}"
        }
    
    async def _evaluate_trading_window(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate trading window rule"""
        # Check if current time is within allowed trading window
        current_time = datetime.now(timezone.utc).time()
        trading_start = rule.parameters.get("trading_start", "09:30")
        trading_end = rule.parameters.get("trading_end", "16:00")
        
        start_time = datetime.strptime(trading_start, "%H:%M").time()
        end_time = datetime.strptime(trading_end, "%H:%M").time()
        
        in_trading_window = start_time <= current_time <= end_time
        
        return {
            "value": 1 if in_trading_window else 0,
            "threshold": 1,
            "unit": "boolean",
            "status": ComplianceStatus.COMPLIANT if in_trading_window else ComplianceStatus.WARNING,
            "description": f"Trading window: {trading_start}-{trading_end}, Current: {current_time}"
        }
    
    async def _evaluate_volume_limit(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate volume limit rule"""
        max_volume = rule.parameters.get("max_daily_volume", 10000000)
        current_volume = 8500000  # Mock current volume
        
        status = ComplianceStatus.COMPLIANT
        if current_volume > max_volume:
            status = ComplianceStatus.VIOLATION
        elif current_volume > max_volume * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_volume,
            "threshold": max_volume,
            "unit": "USD",
            "status": status,
            "description": f"Daily volume: ${current_volume:,.2f}, Limit: ${max_volume:,.2f}"
        }
    
    async def _evaluate_concentration_limit(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate concentration limit rule"""
        max_concentration = rule.parameters.get("max_concentration_percent", 20.0)
        current_concentration = 18.5  # Mock current concentration
        
        status = ComplianceStatus.COMPLIANT
        if current_concentration > max_concentration:
            status = ComplianceStatus.VIOLATION
        elif current_concentration > max_concentration * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_concentration,
            "threshold": max_concentration,
            "unit": "percent",
            "status": status,
            "description": f"Concentration: {current_concentration}%, Limit: {max_concentration}%"
        }
    
    async def _evaluate_documentation(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate documentation rule"""
        required_docs = rule.parameters.get("required_controls", [])
        # Mock documentation check
        documented_controls = len(required_docs) * 0.95  # 95% documented
        
        compliance_rate = documented_controls / len(required_docs) if required_docs else 1.0
        
        status = ComplianceStatus.COMPLIANT
        if compliance_rate < 0.8:
            status = ComplianceStatus.VIOLATION
        elif compliance_rate < 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": compliance_rate,
            "threshold": 1.0,
            "unit": "ratio",
            "status": status,
            "description": f"Documentation compliance: {compliance_rate:.1%}"
        }
    
    async def _evaluate_segregation(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate segregation rule"""
        # Mock segregation check
        segregation_score = 0.92  # 92% segregation compliance
        
        status = ComplianceStatus.COMPLIANT
        if segregation_score < 0.8:
            status = ComplianceStatus.VIOLATION
        elif segregation_score < 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": segregation_score,
            "threshold": 1.0,
            "unit": "ratio",
            "status": status,
            "description": f"Segregation compliance: {segregation_score:.1%}"
        }
    
    async def _evaluate_market_manipulation(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate market manipulation rule"""
        # Mock market manipulation check
        risk_score = 0.15  # Low risk score
        
        status = ComplianceStatus.COMPLIANT
        if risk_score > 0.8:
            status = ComplianceStatus.CRITICAL
        elif risk_score > 0.6:
            status = ComplianceStatus.VIOLATION
        elif risk_score > 0.4:
            status = ComplianceStatus.WARNING
        
        return {
            "value": risk_score,
            "threshold": 0.4,
            "unit": "risk_score",
            "status": status,
            "description": f"Market manipulation risk: {risk_score:.2f}"
        }
    
    async def _evaluate_insider_trading(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate insider trading rule"""
        # Mock insider trading check
        violations_detected = 0
        
        status = ComplianceStatus.COMPLIANT
        if violations_detected > 0:
            status = ComplianceStatus.CRITICAL
        
        return {
            "value": violations_detected,
            "threshold": 0,
            "unit": "count",
            "status": status,
            "description": f"Insider trading violations: {violations_detected}"
        }
    
    async def _evaluate_best_execution(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate best execution rule"""
        quality_threshold = rule.parameters.get("execution_quality_threshold", 0.95)
        current_quality = 0.97  # Mock execution quality
        
        status = ComplianceStatus.COMPLIANT
        if current_quality < quality_threshold:
            status = ComplianceStatus.VIOLATION
        elif current_quality < quality_threshold * 1.05:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_quality,
            "threshold": quality_threshold,
            "unit": "ratio",
            "status": status,
            "description": f"Execution quality: {current_quality:.1%}, Threshold: {quality_threshold:.1%}"
        }
    
    async def _evaluate_liquidity(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate liquidity rule"""
        min_liquidity = rule.parameters.get("min_liquidity_ratio", 0.1)
        current_liquidity = 0.12  # Mock liquidity ratio
        
        status = ComplianceStatus.COMPLIANT
        if current_liquidity < min_liquidity:
            status = ComplianceStatus.VIOLATION
        elif current_liquidity < min_liquidity * 1.1:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_liquidity,
            "threshold": min_liquidity,
            "unit": "ratio",
            "status": status,
            "description": f"Liquidity ratio: {current_liquidity:.1%}, Minimum: {min_liquidity:.1%}"
        }
    
    async def _evaluate_capital_adequacy(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate capital adequacy rule"""
        min_capital_ratio = rule.parameters.get("min_capital_ratio", 0.08)
        current_capital_ratio = 0.12  # Mock capital ratio
        
        status = ComplianceStatus.COMPLIANT
        if current_capital_ratio < min_capital_ratio:
            status = ComplianceStatus.VIOLATION
        elif current_capital_ratio < min_capital_ratio * 1.1:
            status = ComplianceStatus.WARNING
        
        return {
            "value": current_capital_ratio,
            "threshold": min_capital_ratio,
            "unit": "ratio",
            "status": status,
            "description": f"Capital ratio: {current_capital_ratio:.1%}, Minimum: {min_capital_ratio:.1%}"
        }
    
    async def _evaluate_stress_testing(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate stress testing rule"""
        required_frequency = rule.parameters.get("stress_test_frequency", "monthly")
        days_since_last_test = 25  # Mock days since last stress test
        
        frequency_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90
        }
        
        max_days = frequency_days.get(required_frequency, 30)
        
        status = ComplianceStatus.COMPLIANT
        if days_since_last_test > max_days:
            status = ComplianceStatus.VIOLATION
        elif days_since_last_test > max_days * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": days_since_last_test,
            "threshold": max_days,
            "unit": "days",
            "status": status,
            "description": f"Days since stress test: {days_since_last_test}, Required: {required_frequency}"
        }
    
    async def _evaluate_reporting(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate reporting rule"""
        deadline_minutes = rule.parameters.get("reporting_deadline_minutes", 15)
        avg_reporting_time = 12  # Mock average reporting time
        
        status = ComplianceStatus.COMPLIANT
        if avg_reporting_time > deadline_minutes:
            status = ComplianceStatus.VIOLATION
        elif avg_reporting_time > deadline_minutes * 0.9:
            status = ComplianceStatus.WARNING
        
        return {
            "value": avg_reporting_time,
            "threshold": deadline_minutes,
            "unit": "minutes",
            "status": status,
            "description": f"Avg reporting time: {avg_reporting_time} min, Deadline: {deadline_minutes} min"
        }
    
    async def _evaluate_data_protection(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate data protection rule"""
        encryption_required = rule.parameters.get("encryption_required", True)
        access_logging = rule.parameters.get("access_logging", True)
        
        # Mock data protection compliance
        encryption_compliance = 0.98
        logging_compliance = 0.95
        
        overall_compliance = min(encryption_compliance, logging_compliance)
        
        status = ComplianceStatus.COMPLIANT
        if overall_compliance < 0.9:
            status = ComplianceStatus.VIOLATION
        elif overall_compliance < 0.95:
            status = ComplianceStatus.WARNING
        
        return {
            "value": overall_compliance,
            "threshold": 0.95,
            "unit": "ratio",
            "status": status,
            "description": f"Data protection compliance: {overall_compliance:.1%}"
        }
    
    async def _evaluate_retention(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate retention rule"""
        max_retention_days = rule.parameters.get("max_retention_days", 2555)
        records_over_limit = 5  # Mock records over retention limit
        
        status = ComplianceStatus.COMPLIANT
        if records_over_limit > 0:
            status = ComplianceStatus.VIOLATION
        
        return {
            "value": records_over_limit,
            "threshold": 0,
            "unit": "count",
            "status": status,
            "description": f"Records over retention limit: {records_over_limit}"
        }
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.running:
            try:
                # Clean up old metrics (keep last 30 days)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                self.metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
                
                # Clean up resolved violations (keep last 90 days)
                violation_cutoff = datetime.now(timezone.utc) - timedelta(days=90)
                self.violations = [
                    v for v in self.violations 
                    if v.resolved_at is None or v.resolved_at > violation_cutoff
                ]
                
                # Save data
                self._save_data()
                
                # Wait 24 hours before next cleanup
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def add_rule(self, rule: ComplianceRule) -> bool:
        """Add new compliance rule"""
        if rule.rule_id in self.rules:
            return False
        
        self.rules[rule.rule_id] = rule
        self._save_data()
        
        # Start monitoring if system is running
        if self.running and rule.is_active:
            self.monitoring_tasks[rule.rule_id] = asyncio.create_task(
                self._monitor_rule(rule)
            )
        
        await log_audit_event(
            AuditEventType.COMPLIANCE_RULE_ADDED,
            f"Compliance rule added: {rule.name}",
            severity=AuditSeverity.MEDIUM,
            additional_data={
                "rule_id": rule.rule_id,
                "compliance_framework": rule.compliance_framework.value
            }
        )
        
        return True
    
    async def update_rule(self, rule_id: str, **kwargs) -> bool:
        """Update compliance rule"""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        # Update fields
        for field, value in kwargs.items():
            if hasattr(rule, field):
                setattr(rule, field, value)
        
        rule.updated_at = datetime.now(timezone.utc)
        self._save_data()
        
        # Restart monitoring task if needed
        if rule_id in self.monitoring_tasks:
            self.monitoring_tasks[rule_id].cancel()
            if rule.is_active:
                self.monitoring_tasks[rule_id] = asyncio.create_task(
                    self._monitor_rule(rule)
                )
        
        await log_audit_event(
            AuditEventType.COMPLIANCE_RULE_UPDATED,
            f"Compliance rule updated: {rule.name}",
            severity=AuditSeverity.MEDIUM,
            additional_data={"rule_id": rule_id}
        )
        
        return True
    
    async def resolve_violation(self, violation_id: str, resolved_by: str, resolution_notes: str) -> bool:
        """Resolve compliance violation"""
        violation = next((v for v in self.violations if v.violation_id == violation_id), None)
        if not violation:
            return False
        
        violation.resolved_at = datetime.now(timezone.utc)
        violation.resolved_by = resolved_by
        violation.resolution_notes = resolution_notes
        
        self._save_data()
        
        await log_audit_event(
            AuditEventType.COMPLIANCE_VIOLATION_RESOLVED,
            f"Compliance violation resolved: {violation.rule_name}",
            severity=AuditSeverity.MEDIUM,
            additional_data={
                "violation_id": violation_id,
                "resolved_by": resolved_by
            }
        )
        
        return True
    
    def get_compliance_status(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Get overall compliance status"""
        violations = self.violations
        if framework:
            violations = [v for v in violations if v.compliance_framework == framework]
        
        # Filter unresolved violations
        unresolved_violations = [v for v in violations if v.resolved_at is None]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for violation in unresolved_violations:
            severity_counts[violation.severity.value] += 1
        
        # Determine overall status
        overall_status = ComplianceStatus.COMPLIANT
        if severity_counts[ComplianceStatus.CRITICAL.value] > 0:
            overall_status = ComplianceStatus.CRITICAL
        elif severity_counts[ComplianceStatus.VIOLATION.value] > 0:
            overall_status = ComplianceStatus.VIOLATION
        elif severity_counts[ComplianceStatus.WARNING.value] > 0:
            overall_status = ComplianceStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "total_violations": len(unresolved_violations),
            "severity_breakdown": dict(severity_counts),
            "framework": framework.value if framework else "all",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "active_rules": len([r for r in self.rules.values() if r.is_active])
        }
    
    def get_violations(self, 
                      framework: Optional[ComplianceFramework] = None,
                      resolved: Optional[bool] = None,
                      limit: int = 100) -> List[ComplianceViolation]:
        """Get compliance violations"""
        violations = self.violations
        
        # Filter by framework
        if framework:
            violations = [v for v in violations if v.compliance_framework == framework]
        
        # Filter by resolution status
        if resolved is not None:
            if resolved:
                violations = [v for v in violations if v.resolved_at is not None]
            else:
                violations = [v for v in violations if v.resolved_at is None]
        
        # Sort by occurrence date (newest first)
        violations.sort(key=lambda x: x.occurred_at, reverse=True)
        
        return violations[:limit]
    
    def get_metrics(self, 
                   rule_id: Optional[str] = None,
                   hours_back: int = 24) -> List[ComplianceMetric]:
        """Get compliance metrics"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if rule_id:
            metrics = [m for m in metrics if m.rule_id == rule_id]
        
        # Sort by timestamp (newest first)
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        
        return metrics
    
    async def generate_compliance_report(self, 
                                       framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        # Get violations in time range
        violations = [
            v for v in self.violations
            if (v.compliance_framework == framework and
                start_date <= v.occurred_at <= end_date)
        ]
        
        # Get metrics in time range
        metrics = [
            m for m in self.metrics
            if (start_date <= m.timestamp <= end_date and
                self.rules.get(m.rule_id, {}).compliance_framework == framework)
        ]
        
        # Get rules for framework
        framework_rules = [
            r for r in self.rules.values()
            if r.compliance_framework == framework
        ]
        
        # Calculate compliance score
        total_checks = len(metrics)
        compliant_checks = len([m for m in metrics if m.status == ComplianceStatus.COMPLIANT])
        compliance_score = compliant_checks / total_checks if total_checks > 0 else 1.0
        
        # Generate report
        report = {
            "framework": framework.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "compliance_score": compliance_score,
            "total_rules": len(framework_rules),
            "active_rules": len([r for r in framework_rules if r.is_active]),
            "total_violations": len(violations),
            "unresolved_violations": len([v for v in violations if v.resolved_at is None]),
            "violation_breakdown": {
                status.value: len([v for v in violations if v.severity == status])
                for status in ComplianceStatus
            },
            "metrics_summary": {
                "total_checks": total_checks,
                "compliant_checks": compliant_checks,
                "warning_checks": len([m for m in metrics if m.status == ComplianceStatus.WARNING]),
                "violation_checks": len([m for m in metrics if m.status == ComplianceStatus.VIOLATION])
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return report

# Global instance
compliance_monitor = ComplianceMonitor()

# Utility functions
async def start_compliance_monitoring():
    """Start compliance monitoring"""
    await compliance_monitor.start_monitoring()

async def stop_compliance_monitoring():
    """Stop compliance monitoring"""
    await compliance_monitor.stop_monitoring()

async def get_compliance_status(framework: ComplianceFramework = None) -> Dict[str, Any]:
    """Get compliance status"""
    return compliance_monitor.get_compliance_status(framework)

async def add_compliance_rule(rule: ComplianceRule) -> bool:
    """Add compliance rule"""
    return await compliance_monitor.add_rule(rule)

async def resolve_compliance_violation(violation_id: str, resolved_by: str, notes: str) -> bool:
    """Resolve compliance violation"""
    return await compliance_monitor.resolve_violation(violation_id, resolved_by, notes)

async def get_compliance_violations(framework: ComplianceFramework = None) -> List[ComplianceViolation]:
    """Get compliance violations"""
    return compliance_monitor.get_violations(framework=framework, resolved=False)

async def generate_compliance_report(framework: ComplianceFramework, 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, Any]:
    """Generate compliance report"""
    return await compliance_monitor.generate_compliance_report(framework, start_date, end_date)

# Compliance decorators
def compliance_check(framework: ComplianceFramework, rule_type: RuleType):
    """Decorator for compliance checks"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Record compliance check
            await log_audit_event(
                AuditEventType.COMPLIANCE_CHECK_PERFORMED,
                f"Compliance check: {rule_type.value}",
                severity=AuditSeverity.INFO,
                additional_data={
                    "framework": framework.value,
                    "rule_type": rule_type.value,
                    "function": func.__name__
                }
            )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Initialize compliance monitoring
async def initialize_compliance_monitoring():
    """Initialize compliance monitoring system"""
    await start_compliance_monitoring()
    logger.info("Compliance monitoring initialized")

# Shutdown compliance monitoring
async def shutdown_compliance_monitoring():
    """Shutdown compliance monitoring system"""
    await stop_compliance_monitoring()
    logger.info("Compliance monitoring shutdown")
