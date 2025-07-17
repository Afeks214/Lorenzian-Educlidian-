"""
Compliance Monitor for Regulatory Oversight

This module provides comprehensive compliance monitoring, regulatory reporting,
and audit trail functionality for trading system governance.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
import structlog
from pathlib import Path
import sqlite3
import uuid

from ..core.event_bus import EventBus, Event, EventType
from .policy_engine import PolicyEngine, PolicyViolation, PolicySeverity

logger = structlog.get_logger()


class ComplianceStatus(Enum):
    """Status of compliance checks"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    REQUIRES_REVIEW = "requires_review"
    EXEMPTED = "exempted"


class RegulatoryFramework(Enum):
    """Regulatory frameworks to monitor"""
    SEC = "sec"
    FINRA = "finra"
    CFTC = "cftc"
    MiFID2 = "mifid2"
    EMIR = "emir"
    DODD_FRANK = "dodd_frank"
    BASEL_III = "basel_iii"
    COMPANY_POLICY = "company_policy"


class ComplianceRuleType(Enum):
    """Types of compliance rules"""
    POSITION_REPORTING = "position_reporting"
    TRADE_REPORTING = "trade_reporting"
    BEST_EXECUTION = "best_execution"
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"
    RISK_MANAGEMENT = "risk_management"
    CAPITAL_ADEQUACY = "capital_adequacy"
    LIQUIDITY = "liquidity"
    OPERATIONAL_RISK = "operational_risk"


@dataclass
class ComplianceRule:
    """Defines a compliance rule"""
    rule_id: str
    rule_name: str
    rule_type: ComplianceRuleType
    framework: RegulatoryFramework
    description: str
    check_frequency: str  # "real_time", "daily", "weekly", "monthly"
    severity: PolicySeverity
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None
    next_check: Optional[datetime] = None
    violation_count: int = 0


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    violation_id: str
    rule_id: str
    rule_name: str
    framework: RegulatoryFramework
    severity: PolicySeverity
    status: ComplianceStatus
    description: str
    timestamp: datetime
    detected_value: Any
    threshold_value: Any
    context: Dict[str, Any]
    impact_assessment: str
    remediation_required: bool = True
    remediation_deadline: Optional[datetime] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    report_type: str
    framework: RegulatoryFramework
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    total_checks: int
    violations: List[ComplianceViolation]
    compliance_score: float
    summary: Dict[str, Any]
    recommendations: List[str]
    report_data: Dict[str, Any]


class ComplianceChecker:
    """Base class for compliance checkers"""
    
    def __init__(self, rule: ComplianceRule):
        self.rule = rule
        
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check compliance and return violation if any"""
        raise NotImplementedError


class PositionReportingChecker(ComplianceChecker):
    """Checker for position reporting requirements"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check position reporting compliance"""
        positions = context.get("positions", {})
        reporting_threshold = self.rule.parameters.get("reporting_threshold", 5000000)  # $5M
        
        for symbol, position_data in positions.items():
            position_value = abs(position_data.get("market_value", 0))
            
            if position_value > reporting_threshold:
                last_report = position_data.get("last_report_time")
                reporting_frequency = self.rule.parameters.get("reporting_frequency", 24)  # hours
                
                if last_report:
                    hours_since_report = (datetime.now() - last_report).total_seconds() / 3600
                    
                    if hours_since_report > reporting_frequency:
                        return ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            rule_id=self.rule.rule_id,
                            rule_name=self.rule.rule_name,
                            framework=self.rule.framework,
                            severity=self.rule.severity,
                            status=ComplianceStatus.NON_COMPLIANT,
                            description=f"Position in {symbol} worth ${position_value:,.2f} not reported within {reporting_frequency} hours",
                            timestamp=datetime.now(),
                            detected_value=hours_since_report,
                            threshold_value=reporting_frequency,
                            context={"symbol": symbol, "position_value": position_value},
                            impact_assessment="Regulatory reporting violation",
                            remediation_required=True,
                            remediation_deadline=datetime.now() + timedelta(hours=2)
                        )
        
        return None


class TradeReportingChecker(ComplianceChecker):
    """Checker for trade reporting requirements"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check trade reporting compliance"""
        trades = context.get("trades", [])
        reporting_threshold = self.rule.parameters.get("reporting_threshold", 1000000)  # $1M
        max_reporting_delay = self.rule.parameters.get("max_reporting_delay", 1)  # hours
        
        for trade in trades:
            trade_value = abs(trade.get("notional_value", 0))
            
            if trade_value > reporting_threshold:
                trade_time = trade.get("timestamp")
                report_time = trade.get("report_time")
                
                if trade_time and not report_time:
                    hours_since_trade = (datetime.now() - trade_time).total_seconds() / 3600
                    
                    if hours_since_trade > max_reporting_delay:
                        return ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            rule_id=self.rule.rule_id,
                            rule_name=self.rule.rule_name,
                            framework=self.rule.framework,
                            severity=self.rule.severity,
                            status=ComplianceStatus.NON_COMPLIANT,
                            description=f"Trade {trade.get('trade_id')} worth ${trade_value:,.2f} not reported within {max_reporting_delay} hours",
                            timestamp=datetime.now(),
                            detected_value=hours_since_trade,
                            threshold_value=max_reporting_delay,
                            context={"trade_id": trade.get("trade_id"), "trade_value": trade_value},
                            impact_assessment="Trade reporting violation",
                            remediation_required=True,
                            remediation_deadline=datetime.now() + timedelta(hours=1)
                        )
        
        return None


class BestExecutionChecker(ComplianceChecker):
    """Checker for best execution requirements"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check best execution compliance"""
        executions = context.get("executions", [])
        slippage_threshold = self.rule.parameters.get("slippage_threshold", 0.01)  # 1%
        
        for execution in executions:
            expected_price = execution.get("expected_price", 0)
            actual_price = execution.get("actual_price", 0)
            
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                
                if slippage > slippage_threshold:
                    return ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=self.rule.rule_id,
                        rule_name=self.rule.rule_name,
                        framework=self.rule.framework,
                        severity=self.rule.severity,
                        status=ComplianceStatus.NON_COMPLIANT,
                        description=f"Execution {execution.get('execution_id')} had {slippage:.2%} slippage, exceeding {slippage_threshold:.2%} threshold",
                        timestamp=datetime.now(),
                        detected_value=slippage,
                        threshold_value=slippage_threshold,
                        context={"execution_id": execution.get("execution_id"), "slippage": slippage},
                        impact_assessment="Best execution violation",
                        remediation_required=True,
                        remediation_deadline=datetime.now() + timedelta(hours=24)
                    )
        
        return None


class MarketManipulationChecker(ComplianceChecker):
    """Checker for market manipulation detection"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for potential market manipulation"""
        trading_patterns = context.get("trading_patterns", {})
        
        # Check for excessive trading frequency
        trades_per_hour = trading_patterns.get("trades_per_hour", 0)
        max_trades_per_hour = self.rule.parameters.get("max_trades_per_hour", 1000)
        
        if trades_per_hour > max_trades_per_hour:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=self.rule.rule_id,
                rule_name=self.rule.rule_name,
                framework=self.rule.framework,
                severity=self.rule.severity,
                status=ComplianceStatus.REQUIRES_REVIEW,
                description=f"Excessive trading frequency: {trades_per_hour} trades/hour exceeds {max_trades_per_hour} threshold",
                timestamp=datetime.now(),
                detected_value=trades_per_hour,
                threshold_value=max_trades_per_hour,
                context={"trading_patterns": trading_patterns},
                impact_assessment="Potential market manipulation",
                remediation_required=True,
                remediation_deadline=datetime.now() + timedelta(hours=4)
            )
        
        # Check for unusual order patterns
        order_cancel_ratio = trading_patterns.get("order_cancel_ratio", 0)
        max_cancel_ratio = self.rule.parameters.get("max_cancel_ratio", 0.90)
        
        if order_cancel_ratio > max_cancel_ratio:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=self.rule.rule_id,
                rule_name=self.rule.rule_name,
                framework=self.rule.framework,
                severity=self.rule.severity,
                status=ComplianceStatus.REQUIRES_REVIEW,
                description=f"High order cancellation ratio: {order_cancel_ratio:.2%} exceeds {max_cancel_ratio:.2%} threshold",
                timestamp=datetime.now(),
                detected_value=order_cancel_ratio,
                threshold_value=max_cancel_ratio,
                context={"trading_patterns": trading_patterns},
                impact_assessment="Potential spoofing/layering",
                remediation_required=True,
                remediation_deadline=datetime.now() + timedelta(hours=2)
            )
        
        return None


class RiskManagementChecker(ComplianceChecker):
    """Checker for risk management compliance"""
    
    def check_compliance(self, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check risk management compliance"""
        risk_metrics = context.get("risk_metrics", {})
        
        # Check VaR limits
        portfolio_var = risk_metrics.get("portfolio_var", 0)
        var_limit = self.rule.parameters.get("var_limit", 0.05)
        
        if portfolio_var > var_limit:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=self.rule.rule_id,
                rule_name=self.rule.rule_name,
                framework=self.rule.framework,
                severity=self.rule.severity,
                status=ComplianceStatus.NON_COMPLIANT,
                description=f"Portfolio VaR {portfolio_var:.4f} exceeds limit of {var_limit:.4f}",
                timestamp=datetime.now(),
                detected_value=portfolio_var,
                threshold_value=var_limit,
                context={"risk_metrics": risk_metrics},
                impact_assessment="Risk limit breach",
                remediation_required=True,
                remediation_deadline=datetime.now() + timedelta(hours=1)
            )
        
        # Check leverage ratios
        leverage_ratio = risk_metrics.get("leverage_ratio", 0)
        max_leverage = self.rule.parameters.get("max_leverage", 10.0)
        
        if leverage_ratio > max_leverage:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=self.rule.rule_id,
                rule_name=self.rule.rule_name,
                framework=self.rule.framework,
                severity=self.rule.severity,
                status=ComplianceStatus.NON_COMPLIANT,
                description=f"Leverage ratio {leverage_ratio:.2f} exceeds maximum of {max_leverage:.2f}",
                timestamp=datetime.now(),
                detected_value=leverage_ratio,
                threshold_value=max_leverage,
                context={"risk_metrics": risk_metrics},
                impact_assessment="Leverage limit breach",
                remediation_required=True,
                remediation_deadline=datetime.now() + timedelta(minutes=30)
            )
        
        return None


class ComplianceMonitor:
    """Main compliance monitoring system"""
    
    def __init__(self, event_bus: EventBus, policy_engine: PolicyEngine, db_path: str = "compliance.db"):
        self.event_bus = event_bus
        self.policy_engine = policy_engine
        self.db_path = Path(db_path)
        
        # Initialize database
        self._initialize_database()
        
        # Compliance rules and checkers
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.compliance_checkers: Dict[str, ComplianceChecker] = {}
        self.violations: List[ComplianceViolation] = []
        
        # Performance metrics
        self.total_checks = 0
        self.violation_count = 0
        self.compliance_score = 1.0
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Compliance Monitor initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for compliance data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Compliance rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    description TEXT,
                    check_frequency TEXT,
                    severity TEXT,
                    enabled BOOLEAN,
                    parameters TEXT,
                    last_check DATETIME,
                    next_check DATETIME,
                    violation_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Compliance violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    timestamp DATETIME,
                    detected_value TEXT,
                    threshold_value TEXT,
                    context TEXT,
                    impact_assessment TEXT,
                    remediation_required BOOLEAN,
                    remediation_deadline DATETIME,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp DATETIME,
                    resolution_notes TEXT,
                    FOREIGN KEY (rule_id) REFERENCES compliance_rules (rule_id)
                )
            """)
            
            # Compliance reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    period_start DATETIME,
                    period_end DATETIME,
                    generated_at DATETIME,
                    total_checks INTEGER,
                    violation_count INTEGER,
                    compliance_score REAL,
                    summary TEXT,
                    recommendations TEXT,
                    report_data TEXT
                )
            """)
            
            conn.commit()
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        # Position reporting rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="position_reporting_sec",
            rule_name="SEC Position Reporting",
            rule_type=ComplianceRuleType.POSITION_REPORTING,
            framework=RegulatoryFramework.SEC,
            description="Report positions exceeding $5M within 24 hours",
            check_frequency="daily",
            severity=PolicySeverity.HIGH,
            parameters={
                "reporting_threshold": 5000000,
                "reporting_frequency": 24
            }
        ))
        
        # Trade reporting rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="trade_reporting_finra",
            rule_name="FINRA Trade Reporting",
            rule_type=ComplianceRuleType.TRADE_REPORTING,
            framework=RegulatoryFramework.FINRA,
            description="Report trades exceeding $1M within 1 hour",
            check_frequency="real_time",
            severity=PolicySeverity.HIGH,
            parameters={
                "reporting_threshold": 1000000,
                "max_reporting_delay": 1
            }
        ))
        
        # Best execution rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="best_execution_sec",
            rule_name="SEC Best Execution",
            rule_type=ComplianceRuleType.BEST_EXECUTION,
            framework=RegulatoryFramework.SEC,
            description="Monitor execution quality and slippage",
            check_frequency="real_time",
            severity=PolicySeverity.MEDIUM,
            parameters={
                "slippage_threshold": 0.01
            }
        ))
        
        # Market manipulation rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="market_manipulation_cftc",
            rule_name="CFTC Market Manipulation Detection",
            rule_type=ComplianceRuleType.MARKET_MANIPULATION,
            framework=RegulatoryFramework.CFTC,
            description="Detect potential market manipulation patterns",
            check_frequency="real_time",
            severity=PolicySeverity.CRITICAL,
            parameters={
                "max_trades_per_hour": 1000,
                "max_cancel_ratio": 0.90
            }
        ))
        
        # Risk management rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="risk_management_company",
            rule_name="Company Risk Management",
            rule_type=ComplianceRuleType.RISK_MANAGEMENT,
            framework=RegulatoryFramework.COMPANY_POLICY,
            description="Monitor portfolio risk limits",
            check_frequency="real_time",
            severity=PolicySeverity.HIGH,
            parameters={
                "var_limit": 0.05,
                "max_leverage": 10.0
            }
        ))
    
    def _setup_event_handlers(self):
        """Setup event handlers for compliance monitoring"""
        self.event_bus.subscribe(EventType.EXECUTE_TRADE, self._handle_trade_event)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_event)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_event)
        self.event_bus.subscribe(EventType.ORDER_PLACED, self._handle_order_event)
    
    def add_compliance_rule(self, rule: ComplianceRule) -> bool:
        """Add a compliance rule"""
        try:
            self.compliance_rules[rule.rule_id] = rule
            
            # Create appropriate checker
            checker = self._create_checker(rule)
            if checker:
                self.compliance_checkers[rule.rule_id] = checker
            
            # Store in database
            self._store_rule_in_db(rule)
            
            logger.info("Compliance rule added", rule_id=rule.rule_id)
            return True
            
        except Exception as e:
            logger.error("Failed to add compliance rule", rule_id=rule.rule_id, error=str(e))
            return False
    
    def _create_checker(self, rule: ComplianceRule) -> Optional[ComplianceChecker]:
        """Create appropriate checker for rule type"""
        if rule.rule_type == ComplianceRuleType.POSITION_REPORTING:
            return PositionReportingChecker(rule)
        elif rule.rule_type == ComplianceRuleType.TRADE_REPORTING:
            return TradeReportingChecker(rule)
        elif rule.rule_type == ComplianceRuleType.BEST_EXECUTION:
            return BestExecutionChecker(rule)
        elif rule.rule_type == ComplianceRuleType.MARKET_MANIPULATION:
            return MarketManipulationChecker(rule)
        elif rule.rule_type == ComplianceRuleType.RISK_MANAGEMENT:
            return RiskManagementChecker(rule)
        else:
            logger.warning("Unknown compliance rule type", rule_type=rule.rule_type)
            return None
    
    def _store_rule_in_db(self, rule: ComplianceRule):
        """Store compliance rule in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO compliance_rules (
                    rule_id, rule_name, rule_type, framework, description,
                    check_frequency, severity, enabled, parameters, last_check,
                    next_check, violation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.rule_name,
                rule.rule_type.value,
                rule.framework.value,
                rule.description,
                rule.check_frequency,
                rule.severity.value,
                rule.enabled,
                json.dumps(rule.parameters),
                rule.last_check,
                rule.next_check,
                rule.violation_count
            ))
            conn.commit()
    
    def check_compliance(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check compliance against all applicable rules"""
        violations = []
        self.total_checks += 1
        
        for rule_id, checker in self.compliance_checkers.items():
            rule = self.compliance_rules[rule_id]
            
            if not rule.enabled:
                continue
            
            try:
                violation = checker.check_compliance(context)
                
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)
                    self.violation_count += 1
                    
                    # Update rule statistics
                    rule.violation_count += 1
                    rule.last_check = datetime.now()
                    
                    # Store violation in database
                    self._store_violation_in_db(violation)
                    
                    # Publish violation event
                    self._publish_violation_event(violation)
                    
                    logger.warning(
                        "Compliance violation detected",
                        rule_id=rule_id,
                        violation_id=violation.violation_id,
                        severity=violation.severity.value
                    )
                
            except Exception as e:
                logger.error("Error checking compliance rule", rule_id=rule_id, error=str(e))
                continue
        
        # Update compliance score
        self._update_compliance_score()
        
        return violations
    
    def _store_violation_in_db(self, violation: ComplianceViolation):
        """Store compliance violation in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO compliance_violations (
                    violation_id, rule_id, rule_name, framework, severity,
                    status, description, timestamp, detected_value, threshold_value,
                    context, impact_assessment, remediation_required, remediation_deadline,
                    resolved, resolution_timestamp, resolution_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.violation_id,
                violation.rule_id,
                violation.rule_name,
                violation.framework.value,
                violation.severity.value,
                violation.status.value,
                violation.description,
                violation.timestamp,
                json.dumps(violation.detected_value),
                json.dumps(violation.threshold_value),
                json.dumps(violation.context),
                violation.impact_assessment,
                violation.remediation_required,
                violation.remediation_deadline,
                violation.resolved,
                violation.resolution_timestamp,
                violation.resolution_notes
            ))
            conn.commit()
    
    def _publish_violation_event(self, violation: ComplianceViolation):
        """Publish compliance violation event"""
        event = self.event_bus.create_event(
            event_type=EventType.RISK_BREACH,
            payload={
                "type": "compliance_violation",
                "violation_id": violation.violation_id,
                "rule_id": violation.rule_id,
                "framework": violation.framework.value,
                "severity": violation.severity.value,
                "status": violation.status.value,
                "description": violation.description,
                "timestamp": violation.timestamp.isoformat(),
                "remediation_required": violation.remediation_required,
                "remediation_deadline": violation.remediation_deadline.isoformat() if violation.remediation_deadline else None
            },
            source="compliance_monitor"
        )
        
        self.event_bus.publish(event)
    
    def _update_compliance_score(self):
        """Update overall compliance score"""
        if self.total_checks == 0:
            self.compliance_score = 1.0
        else:
            # Calculate score based on violation rate and severity
            violation_rate = self.violation_count / self.total_checks
            
            # Weight violations by severity
            severity_weights = {
                PolicySeverity.LOW: 0.25,
                PolicySeverity.MEDIUM: 0.5,
                PolicySeverity.HIGH: 0.75,
                PolicySeverity.CRITICAL: 1.0
            }
            
            weighted_violations = sum(
                severity_weights.get(v.severity, 1.0) 
                for v in self.violations if not v.resolved
            )
            
            # Calculate score (0-1 range)
            if weighted_violations == 0:
                self.compliance_score = 1.0
            else:
                self.compliance_score = max(0.0, 1.0 - (weighted_violations / self.total_checks))
    
    def _handle_trade_event(self, event: Event):
        """Handle trade execution events"""
        try:
            payload = event.payload
            
            context = {
                "trades": [payload],
                "executions": [payload],
                "timestamp": datetime.now(),
                "event_type": "trade_execution"
            }
            
            violations = self.check_compliance(context)
            
            # Handle violations
            for violation in violations:
                self._handle_violation(violation)
                
        except Exception as e:
            logger.error("Error handling trade event for compliance", error=str(e))
    
    def _handle_position_event(self, event: Event):
        """Handle position update events"""
        try:
            payload = event.payload
            
            context = {
                "positions": {payload.get("symbol", ""): payload},
                "timestamp": datetime.now(),
                "event_type": "position_update"
            }
            
            violations = self.check_compliance(context)
            
            # Handle violations
            for violation in violations:
                self._handle_violation(violation)
                
        except Exception as e:
            logger.error("Error handling position event for compliance", error=str(e))
    
    def _handle_risk_event(self, event: Event):
        """Handle risk breach events"""
        try:
            payload = event.payload
            
            context = {
                "risk_metrics": payload,
                "timestamp": datetime.now(),
                "event_type": "risk_breach"
            }
            
            violations = self.check_compliance(context)
            
            # Handle violations
            for violation in violations:
                self._handle_violation(violation)
                
        except Exception as e:
            logger.error("Error handling risk event for compliance", error=str(e))
    
    def _handle_order_event(self, event: Event):
        """Handle order placement events"""
        try:
            payload = event.payload
            
            # Build trading patterns context
            context = {
                "trading_patterns": {
                    "trades_per_hour": payload.get("trades_per_hour", 0),
                    "order_cancel_ratio": payload.get("cancel_ratio", 0),
                    "order_volume": payload.get("volume", 0)
                },
                "timestamp": datetime.now(),
                "event_type": "order_placed"
            }
            
            violations = self.check_compliance(context)
            
            # Handle violations
            for violation in violations:
                self._handle_violation(violation)
                
        except Exception as e:
            logger.error("Error handling order event for compliance", error=str(e))
    
    def _handle_violation(self, violation: ComplianceViolation):
        """Handle compliance violation"""
        try:
            if violation.severity == PolicySeverity.CRITICAL:
                logger.critical("Critical compliance violation", violation_id=violation.violation_id)
                # Could trigger emergency protocols
            
            elif violation.severity == PolicySeverity.HIGH:
                logger.error("High severity compliance violation", violation_id=violation.violation_id)
                # Could trigger automatic remediation
            
            elif violation.severity == PolicySeverity.MEDIUM:
                logger.warning("Medium severity compliance violation", violation_id=violation.violation_id)
                # Could trigger notifications
            
            else:
                logger.info("Low severity compliance violation", violation_id=violation.violation_id)
                # Could trigger logging only
                
        except Exception as e:
            logger.error("Error handling compliance violation", violation_id=violation.violation_id, error=str(e))
    
    def generate_compliance_report(
        self,
        framework: Optional[RegulatoryFramework] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        if period_end is None:
            period_end = datetime.now()
        if period_start is None:
            period_start = period_end - timedelta(days=30)
        
        # Filter violations by period and framework
        filtered_violations = [
            v for v in self.violations
            if period_start <= v.timestamp <= period_end
            and (framework is None or v.framework == framework)
        ]
        
        # Calculate statistics
        total_violations = len(filtered_violations)
        resolved_violations = len([v for v in filtered_violations if v.resolved])
        critical_violations = len([v for v in filtered_violations if v.severity == PolicySeverity.CRITICAL])
        high_violations = len([v for v in filtered_violations if v.severity == PolicySeverity.HIGH])
        
        # Generate summary
        summary = {
            "total_violations": total_violations,
            "resolved_violations": resolved_violations,
            "unresolved_violations": total_violations - resolved_violations,
            "critical_violations": critical_violations,
            "high_violations": high_violations,
            "resolution_rate": resolved_violations / total_violations if total_violations > 0 else 1.0,
            "compliance_score": self.compliance_score
        }
        
        # Generate recommendations
        recommendations = []
        if critical_violations > 0:
            recommendations.append("Immediately address critical compliance violations")
        if high_violations > 3:
            recommendations.append("Review and strengthen risk controls")
        if summary["resolution_rate"] < 0.8:
            recommendations.append("Improve violation resolution processes")
        
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            report_type="compliance_summary",
            framework=framework or RegulatoryFramework.COMPANY_POLICY,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(),
            total_checks=self.total_checks,
            violations=filtered_violations,
            compliance_score=self.compliance_score,
            summary=summary,
            recommendations=recommendations,
            report_data={}
        )
        
        # Store report in database
        self._store_report_in_db(report)
        
        return report
    
    def _store_report_in_db(self, report: ComplianceReport):
        """Store compliance report in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO compliance_reports (
                    report_id, report_type, framework, period_start, period_end,
                    generated_at, total_checks, violation_count, compliance_score,
                    summary, recommendations, report_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.report_type,
                report.framework.value,
                report.period_start,
                report.period_end,
                report.generated_at,
                report.total_checks,
                len(report.violations),
                report.compliance_score,
                json.dumps(report.summary),
                json.dumps(report.recommendations),
                json.dumps(report.report_data)
            ))
            conn.commit()
    
    def resolve_violation(self, violation_id: str, resolution_notes: str = "") -> bool:
        """Resolve a compliance violation"""
        try:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution_timestamp = datetime.now()
                    violation.resolution_notes = resolution_notes
                    
                    # Update in database
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE compliance_violations 
                            SET resolved = TRUE, resolution_timestamp = ?, resolution_notes = ?
                            WHERE violation_id = ?
                        """, (violation.resolution_timestamp, resolution_notes, violation_id))
                        conn.commit()
                    
                    # Update compliance score
                    self._update_compliance_score()
                    
                    logger.info("Compliance violation resolved", violation_id=violation_id)
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error resolving compliance violation", violation_id=violation_id, error=str(e))
            return False
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        unresolved_violations = [v for v in self.violations if not v.resolved]
        
        return {
            "compliance_score": self.compliance_score,
            "total_checks": self.total_checks,
            "total_violations": self.violation_count,
            "unresolved_violations": len(unresolved_violations),
            "active_rules": len([r for r in self.compliance_rules.values() if r.enabled]),
            "critical_violations": len([v for v in unresolved_violations if v.severity == PolicySeverity.CRITICAL]),
            "high_violations": len([v for v in unresolved_violations if v.severity == PolicySeverity.HIGH]),
            "last_check": max([r.last_check for r in self.compliance_rules.values() if r.last_check], default=None)
        }
    
    def get_violations(
        self,
        framework: Optional[RegulatoryFramework] = None,
        severity: Optional[PolicySeverity] = None,
        resolved: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[ComplianceViolation]:
        """Get violations with optional filtering"""
        violations = self.violations.copy()
        
        # Apply filters
        if framework:
            violations = [v for v in violations if v.framework == framework]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]
        
        # Sort by timestamp (newest first)
        violations.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            violations = violations[:limit]
        
        return violations
    
    def cleanup_old_data(self, days_old: int = 90):
        """Clean up old compliance data"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean up old resolved violations
            cursor.execute("""
                DELETE FROM compliance_violations 
                WHERE resolved = TRUE AND resolution_timestamp < ?
            """, (cutoff_date,))
            
            # Clean up old reports
            cursor.execute("""
                DELETE FROM compliance_reports 
                WHERE generated_at < ?
            """, (cutoff_date,))
            
            conn.commit()
            
            logger.info("Cleaned up old compliance data", cutoff_date=cutoff_date)