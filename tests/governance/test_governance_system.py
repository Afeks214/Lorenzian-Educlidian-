"""
Comprehensive Test Suite for Governance System

This test suite provides extensive testing for the governance system including
policy enforcement, compliance validation, audit trails, and regulatory reporting.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import shutil
from pathlib import Path
import sqlite3
import pandas as pd
import uuid

# Import governance components
from src.governance.policy_engine import (
    PolicyEngine, PolicyRule, PolicyViolation, PolicyContext,
    PolicyType, PolicySeverity, PolicyAction, PositionLimitPolicy,
    RiskLimitPolicy, TradingHoursPolicy, ConcentrationLimitPolicy,
    DrawdownLimitPolicy, CustomFunctionPolicy
)
from src.governance.compliance_monitor import (
    ComplianceMonitor, ComplianceRule, ComplianceViolation, ComplianceReport,
    ComplianceStatus, RegulatoryFramework, ComplianceRuleType,
    PositionReportingChecker, TradeReportingChecker, BestExecutionChecker,
    MarketManipulationChecker, RiskManagementChecker
)
from src.governance.audit_system import (
    AuditSystem, AuditEvent, AuditTrail, AuditContext, AuditStorage,
    AuditEventType, AuditSeverity, AuditOutcome
)
from src.governance.regulatory_reporter import (
    RegulatoryReporter, RegulatoryReport, ReportTemplate, ReportGenerator,
    ReportType, ReportFormat, ReportFrequency, ReportValidator,
    TradeReportingTemplate, PositionReportingTemplate, RiskReportingTemplate
)
from src.core.event_bus import EventBus, EventType


class TestPolicyEngine:
    """Test the policy enforcement engine"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def policy_engine(self, event_bus):
        """Create policy engine for testing"""
        return PolicyEngine(event_bus)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample policy context"""
        return PolicyContext(
            timestamp=datetime.now(),
            user_id="test_user",
            system_component="trading_engine",
            action_type="execute_trade",
            parameters={
                "symbol": "ETH-USD",
                "position_size": 5000.0,
                "price": 2000.0
            },
            current_state={
                "portfolio_risk": {"var_95": 0.03},
                "positions": {"ETH-USD": {"value": 100000.0}}
            },
            portfolio_data={
                "total_value": 1000000.0,
                "current_drawdown": 0.10,
                "positions": {
                    "ETH-USD": {"value": 100000.0}
                }
            }
        )
    
    def test_policy_engine_initialization(self, policy_engine):
        """Test policy engine initialization"""
        assert policy_engine is not None
        assert policy_engine.enabled is True
        assert len(policy_engine.policies) > 0  # Should have default policies
        assert policy_engine.evaluation_count == 0
        assert policy_engine.violation_count == 0
    
    def test_position_limit_policy(self, sample_context):
        """Test position limit policy"""
        policy = PositionLimitPolicy(
            policy_id="test_position_limit",
            max_position_size=3000.0,
            symbol="ETH-USD",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.BLOCK
        )
        
        # Test violation
        violation = policy.evaluate(sample_context)
        assert violation is not None
        assert violation.policy_id == "test_position_limit"
        assert violation.severity == PolicySeverity.HIGH
        assert violation.suggested_action == PolicyAction.BLOCK
        assert violation.violation_value == 5000.0
        assert violation.threshold_value == 3000.0
    
    def test_risk_limit_policy(self, sample_context):
        """Test risk limit policy"""
        policy = RiskLimitPolicy(
            policy_id="test_risk_limit",
            max_portfolio_risk=0.02,
            risk_measure="var_95",
            severity=PolicySeverity.CRITICAL,
            action=PolicyAction.REDUCE
        )
        
        # Test violation
        violation = policy.evaluate(sample_context)
        assert violation is not None
        assert violation.policy_id == "test_risk_limit"
        assert violation.severity == PolicySeverity.CRITICAL
        assert violation.suggested_action == PolicyAction.REDUCE
        assert violation.violation_value == 0.03
        assert violation.threshold_value == 0.02
    
    def test_trading_hours_policy(self, sample_context):
        """Test trading hours policy"""
        policy = TradingHoursPolicy(
            policy_id="test_trading_hours",
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16],
            severity=PolicySeverity.MEDIUM,
            action=PolicyAction.BLOCK
        )
        
        # Test during allowed hours
        allowed_context = sample_context
        allowed_context.timestamp = datetime.now().replace(hour=10)
        violation = policy.evaluate(allowed_context)
        assert violation is None
        
        # Test outside allowed hours
        blocked_context = sample_context
        blocked_context.timestamp = datetime.now().replace(hour=20)
        violation = policy.evaluate(blocked_context)
        assert violation is not None
        assert violation.policy_id == "test_trading_hours"
    
    def test_concentration_limit_policy(self, sample_context):
        """Test concentration limit policy"""
        policy = ConcentrationLimitPolicy(
            policy_id="test_concentration",
            max_concentration=0.05,  # 5% max
            concentration_type="symbol",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.WARN
        )
        
        # Test violation (10% concentration)
        violation = policy.evaluate(sample_context)
        assert violation is not None
        assert violation.policy_id == "test_concentration"
        assert violation.severity == PolicySeverity.HIGH
        assert violation.suggested_action == PolicyAction.WARN
        assert violation.violation_value == 0.1  # 100k / 1M = 10%
        assert violation.threshold_value == 0.05
    
    def test_drawdown_limit_policy(self, sample_context):
        """Test drawdown limit policy"""
        policy = DrawdownLimitPolicy(
            policy_id="test_drawdown",
            max_drawdown=0.05,  # 5% max
            lookback_period=30,
            severity=PolicySeverity.CRITICAL,
            action=PolicyAction.TERMINATE
        )
        
        # Test violation (10% drawdown)
        violation = policy.evaluate(sample_context)
        assert violation is not None
        assert violation.policy_id == "test_drawdown"
        assert violation.severity == PolicySeverity.CRITICAL
        assert violation.suggested_action == PolicyAction.TERMINATE
        assert violation.violation_value == 0.10
        assert violation.threshold_value == 0.05
    
    def test_custom_function_policy(self, sample_context):
        """Test custom function policy"""
        def custom_check(context: PolicyContext) -> PolicyViolation:
            """Custom policy check"""
            if context.parameters.get("position_size", 0) > 1000:
                return PolicyViolation(
                    policy_id="custom_policy",
                    policy_name="Custom Policy",
                    severity=PolicySeverity.LOW,
                    message="Custom violation detected",
                    context=context,
                    timestamp=datetime.now(),
                    violation_value=context.parameters.get("position_size", 0),
                    threshold_value=1000,
                    suggested_action=PolicyAction.WARN
                )
            return None
        
        policy = CustomFunctionPolicy(
            policy_id="custom_policy",
            policy_name="Custom Policy",
            evaluation_function=custom_check,
            severity=PolicySeverity.LOW,
            action=PolicyAction.WARN
        )
        
        # Test violation
        violation = policy.evaluate(sample_context)
        assert violation is not None
        assert violation.policy_id == "custom_policy"
        assert violation.severity == PolicySeverity.LOW
    
    def test_policy_engine_evaluation(self, policy_engine, sample_context):
        """Test policy engine evaluation"""
        # Add custom policy
        policy = PositionLimitPolicy(
            policy_id="test_eval_policy",
            max_position_size=1000.0,
            symbol="ETH-USD",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.BLOCK
        )
        
        policy_engine.add_policy(policy)
        
        # Evaluate policies
        violations = policy_engine.evaluate_policies(sample_context)
        
        # Should have violations from default policies and our custom policy
        assert len(violations) > 0
        assert policy_engine.evaluation_count > 0
        assert policy_engine.violation_count > 0
    
    def test_policy_management(self, policy_engine):
        """Test policy management operations"""
        # Test adding policy
        policy = PositionLimitPolicy(
            policy_id="test_mgmt_policy",
            max_position_size=1000.0,
            symbol="BTC-USD",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.BLOCK
        )
        
        success = policy_engine.add_policy(policy)
        assert success is True
        assert "test_mgmt_policy" in policy_engine.policies
        
        # Test enabling/disabling policy
        success = policy_engine.disable_policy("test_mgmt_policy")
        assert success is True
        assert policy_engine.policies["test_mgmt_policy"].enabled is False
        
        success = policy_engine.enable_policy("test_mgmt_policy")
        assert success is True
        assert policy_engine.policies["test_mgmt_policy"].enabled is True
        
        # Test removing policy
        success = policy_engine.remove_policy("test_mgmt_policy")
        assert success is True
        assert "test_mgmt_policy" not in policy_engine.policies
    
    def test_violation_resolution(self, policy_engine, sample_context):
        """Test violation resolution"""
        # Create violation
        violations = policy_engine.evaluate_policies(sample_context)
        
        if violations:
            violation = violations[0]
            
            # Resolve violation
            success = policy_engine.resolve_violation(
                violation.violation_id,
                "Test resolution"
            )
            assert success is True
            assert violation.resolved is True
            assert violation.resolution_notes == "Test resolution"
            assert violation.resolution_timestamp is not None
    
    def test_policy_status_reporting(self, policy_engine):
        """Test policy status reporting"""
        # Get all policies status
        all_status = policy_engine.get_all_policies_status()
        assert isinstance(all_status, dict)
        assert len(all_status) > 0
        
        # Get specific policy status
        policy_ids = list(policy_engine.policies.keys())
        if policy_ids:
            policy_id = policy_ids[0]
            status = policy_engine.get_policy_status(policy_id)
            assert status is not None
            assert status["policy_id"] == policy_id
            assert "enabled" in status
            assert "severity" in status
    
    def test_engine_status(self, policy_engine):
        """Test engine status reporting"""
        status = policy_engine.get_engine_status()
        
        assert "enabled" in status
        assert "total_policies" in status
        assert "active_policies" in status
        assert "total_evaluations" in status
        assert "total_violations" in status
        assert "unresolved_violations" in status
        assert status["enabled"] is True
        assert status["total_policies"] > 0


class TestComplianceMonitor:
    """Test the compliance monitoring system"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_compliance.db"
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def policy_engine(self, event_bus):
        """Create policy engine for testing"""
        return PolicyEngine(event_bus)
    
    @pytest.fixture
    def compliance_monitor(self, event_bus, policy_engine, temp_db_path):
        """Create compliance monitor for testing"""
        return ComplianceMonitor(event_bus, policy_engine, temp_db_path)
    
    def test_compliance_monitor_initialization(self, compliance_monitor):
        """Test compliance monitor initialization"""
        assert compliance_monitor is not None
        assert len(compliance_monitor.compliance_rules) > 0
        assert len(compliance_monitor.compliance_checkers) > 0
        assert compliance_monitor.db_path.exists()
    
    def test_position_reporting_checker(self):
        """Test position reporting compliance checker"""
        rule = ComplianceRule(
            rule_id="test_position_reporting",
            rule_name="Test Position Reporting",
            rule_type=ComplianceRuleType.POSITION_REPORTING,
            framework=RegulatoryFramework.SEC,
            description="Test position reporting rule",
            check_frequency="daily",
            severity=PolicySeverity.HIGH,
            parameters={
                "reporting_threshold": 1000000,
                "reporting_frequency": 24
            }
        )
        
        checker = PositionReportingChecker(rule)
        
        # Test violation (position not reported)
        context = {
            "positions": {
                "ETH-USD": {
                    "market_value": 2000000,
                    "last_report_time": datetime.now() - timedelta(hours=30)
                }
            }
        }
        
        violation = checker.check_compliance(context)
        assert violation is not None
        assert violation.rule_id == "test_position_reporting"
        assert violation.framework == RegulatoryFramework.SEC
        assert violation.severity == PolicySeverity.HIGH
    
    def test_trade_reporting_checker(self):
        """Test trade reporting compliance checker"""
        rule = ComplianceRule(
            rule_id="test_trade_reporting",
            rule_name="Test Trade Reporting",
            rule_type=ComplianceRuleType.TRADE_REPORTING,
            framework=RegulatoryFramework.FINRA,
            description="Test trade reporting rule",
            check_frequency="real_time",
            severity=PolicySeverity.HIGH,
            parameters={
                "reporting_threshold": 500000,
                "max_reporting_delay": 1
            }
        )
        
        checker = TradeReportingChecker(rule)
        
        # Test violation (trade not reported)
        context = {
            "trades": [{
                "trade_id": "TRD001",
                "notional_value": 1000000,
                "timestamp": datetime.now() - timedelta(hours=2),
                "report_time": None
            }]
        }
        
        violation = checker.check_compliance(context)
        assert violation is not None
        assert violation.rule_id == "test_trade_reporting"
        assert violation.framework == RegulatoryFramework.FINRA
    
    def test_best_execution_checker(self):
        """Test best execution compliance checker"""
        rule = ComplianceRule(
            rule_id="test_best_execution",
            rule_name="Test Best Execution",
            rule_type=ComplianceRuleType.BEST_EXECUTION,
            framework=RegulatoryFramework.SEC,
            description="Test best execution rule",
            check_frequency="real_time",
            severity=PolicySeverity.MEDIUM,
            parameters={
                "slippage_threshold": 0.005  # 0.5%
            }
        )
        
        checker = BestExecutionChecker(rule)
        
        # Test violation (excessive slippage)
        context = {
            "executions": [{
                "execution_id": "EXE001",
                "expected_price": 2000.0,
                "actual_price": 2030.0  # 1.5% slippage
            }]
        }
        
        violation = checker.check_compliance(context)
        assert violation is not None
        assert violation.rule_id == "test_best_execution"
        assert violation.framework == RegulatoryFramework.SEC
    
    def test_market_manipulation_checker(self):
        """Test market manipulation compliance checker"""
        rule = ComplianceRule(
            rule_id="test_market_manipulation",
            rule_name="Test Market Manipulation",
            rule_type=ComplianceRuleType.MARKET_MANIPULATION,
            framework=RegulatoryFramework.CFTC,
            description="Test market manipulation rule",
            check_frequency="real_time",
            severity=PolicySeverity.CRITICAL,
            parameters={
                "max_trades_per_hour": 500,
                "max_cancel_ratio": 0.80
            }
        )
        
        checker = MarketManipulationChecker(rule)
        
        # Test violation (excessive trading)
        context = {
            "trading_patterns": {
                "trades_per_hour": 800,
                "order_cancel_ratio": 0.95
            }
        }
        
        violation = checker.check_compliance(context)
        assert violation is not None
        assert violation.rule_id == "test_market_manipulation"
        assert violation.framework == RegulatoryFramework.CFTC
        assert violation.severity == PolicySeverity.CRITICAL
    
    def test_risk_management_checker(self):
        """Test risk management compliance checker"""
        rule = ComplianceRule(
            rule_id="test_risk_management",
            rule_name="Test Risk Management",
            rule_type=ComplianceRuleType.RISK_MANAGEMENT,
            framework=RegulatoryFramework.COMPANY_POLICY,
            description="Test risk management rule",
            check_frequency="real_time",
            severity=PolicySeverity.HIGH,
            parameters={
                "var_limit": 0.03,
                "max_leverage": 5.0
            }
        )
        
        checker = RiskManagementChecker(rule)
        
        # Test violation (excessive risk)
        context = {
            "risk_metrics": {
                "portfolio_var": 0.05,
                "leverage_ratio": 8.0
            }
        }
        
        violation = checker.check_compliance(context)
        assert violation is not None
        assert violation.rule_id == "test_risk_management"
        assert violation.framework == RegulatoryFramework.COMPANY_POLICY
    
    def test_compliance_check_integration(self, compliance_monitor):
        """Test integrated compliance checking"""
        # Test context with multiple potential violations
        context = {
            "positions": {
                "ETH-USD": {
                    "market_value": 6000000,
                    "last_report_time": datetime.now() - timedelta(hours=30)
                }
            },
            "trades": [{
                "trade_id": "TRD001",
                "notional_value": 2000000,
                "timestamp": datetime.now() - timedelta(hours=2),
                "report_time": None
            }],
            "trading_patterns": {
                "trades_per_hour": 1200,
                "order_cancel_ratio": 0.95
            },
            "risk_metrics": {
                "portfolio_var": 0.06,
                "leverage_ratio": 12.0
            }
        }
        
        violations = compliance_monitor.check_compliance(context)
        
        # Should detect multiple violations
        assert len(violations) > 0
        assert compliance_monitor.total_checks > 0
        assert compliance_monitor.violation_count > 0
    
    def test_compliance_reporting(self, compliance_monitor):
        """Test compliance report generation"""
        # Generate some violations first
        context = {
            "risk_metrics": {
                "portfolio_var": 0.08,
                "leverage_ratio": 15.0
            }
        }
        
        violations = compliance_monitor.check_compliance(context)
        
        # Generate report
        report = compliance_monitor.generate_compliance_report(
            framework=RegulatoryFramework.COMPANY_POLICY,
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now()
        )
        
        assert report is not None
        assert report.framework == RegulatoryFramework.COMPANY_POLICY
        assert report.total_checks > 0
        assert len(report.violations) > 0
        assert report.compliance_score <= 1.0
        assert len(report.summary) > 0
    
    def test_violation_resolution(self, compliance_monitor):
        """Test violation resolution"""
        # Create violation
        context = {
            "risk_metrics": {
                "portfolio_var": 0.08
            }
        }
        
        violations = compliance_monitor.check_compliance(context)
        
        if violations:
            violation = violations[0]
            
            # Resolve violation
            success = compliance_monitor.resolve_violation(
                violation.violation_id,
                "Risk reduced to acceptable levels"
            )
            
            assert success is True
            assert violation.resolved is True
            assert violation.resolution_notes == "Risk reduced to acceptable levels"
    
    def test_compliance_status(self, compliance_monitor):
        """Test compliance status reporting"""
        status = compliance_monitor.get_compliance_status()
        
        assert "compliance_score" in status
        assert "total_checks" in status
        assert "total_violations" in status
        assert "unresolved_violations" in status
        assert "active_rules" in status
        assert status["compliance_score"] <= 1.0
        assert status["total_checks"] >= 0


class TestAuditSystem:
    """Test the audit system"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_audit.db"
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def audit_storage(self, temp_db_path):
        """Create audit storage for testing"""
        return AuditStorage(temp_db_path)
    
    @pytest.fixture
    def audit_system(self, event_bus, audit_storage):
        """Create audit system for testing"""
        return AuditSystem(event_bus, audit_storage)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample audit context"""
        return AuditContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100",
            user_agent="test_agent",
            system_component="trading_engine",
            action="execute_trade",
            resource="ETH-USD",
            additional_data={"trade_id": "TRD001"}
        )
    
    def test_audit_storage_initialization(self, audit_storage):
        """Test audit storage initialization"""
        assert audit_storage is not None
        assert audit_storage.db_path.exists()
        
        # Check database schema
        with sqlite3.connect(audit_storage.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "audit_events" in tables
            assert "audit_trails" in tables
            assert "audit_trail_events" in tables
            assert "audit_summaries" in tables
    
    def test_audit_event_creation(self, sample_context):
        """Test audit event creation and integrity"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.TRADE_EXECUTED,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            timestamp=datetime.now(),
            context=sample_context,
            description="Test trade execution",
            details={"symbol": "ETH-USD", "quantity": 100}
        )
        
        assert event.event_id is not None
        assert event.event_type == AuditEventType.TRADE_EXECUTED
        assert event.severity == AuditSeverity.INFO
        assert event.outcome == AuditOutcome.SUCCESS
        assert event.checksum is not None
        assert len(event.checksum) == 64  # SHA-256 hash length
        
        # Test integrity verification
        assert event.verify_integrity() is True
        
        # Test tampering detection
        event.description = "Modified description"
        assert event.verify_integrity() is False
    
    def test_audit_event_storage(self, audit_storage, sample_context):
        """Test audit event storage and retrieval"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.TRADE_EXECUTED,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            timestamp=datetime.now(),
            context=sample_context,
            description="Test trade execution",
            details={"symbol": "ETH-USD", "quantity": 100}
        )
        
        # Store event
        success = audit_storage.store_event(event)
        assert success is True
        
        # Retrieve events
        events = audit_storage.get_events(limit=10)
        assert len(events) > 0
        
        retrieved_event = events[0]
        assert retrieved_event.event_id == event.event_id
        assert retrieved_event.event_type == event.event_type
        assert retrieved_event.verify_integrity() is True
    
    def test_audit_trail_management(self, audit_storage, sample_context):
        """Test audit trail creation and management"""
        trail = AuditTrail(
            trail_id=str(uuid.uuid4()),
            trail_name="Test Trading Session",
            start_time=datetime.now(),
            end_time=None,
            events=[],
            metadata={"session_type": "automated"}
        )
        
        # Add events to trail
        for i in range(3):
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                timestamp=datetime.now(),
                context=sample_context,
                description=f"Trade {i+1}",
                details={"trade_number": i+1}
            )
            trail.add_event(event)
        
        # Verify chain integrity
        assert trail.verify_chain_integrity() is True
        
        # Store trail
        success = audit_storage.store_trail(trail)
        assert success is True
        
        # Retrieve trail
        retrieved_trail = audit_storage.get_trail(trail.trail_id)
        assert retrieved_trail is not None
        assert retrieved_trail.trail_id == trail.trail_id
        assert len(retrieved_trail.events) == 3
        assert retrieved_trail.verify_chain_integrity() is True
    
    def test_audit_system_event_logging(self, audit_system, sample_context):
        """Test audit system event logging"""
        event_id = audit_system.log_event(
            event_type=AuditEventType.TRADE_EXECUTED,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            context=sample_context,
            description="Test trade execution",
            details={"symbol": "ETH-USD", "quantity": 100}
        )
        
        assert event_id is not None
        assert len(event_id) > 0
        assert audit_system.event_count > 0
        
        # Verify event was stored
        events = audit_system.storage.get_events(limit=1)
        assert len(events) > 0
        assert events[0].event_id == event_id
    
    def test_audit_trail_workflow(self, audit_system, sample_context):
        """Test audit trail workflow"""
        # Start trail
        trail_id = audit_system.start_trail(
            "Test Trading Session",
            {"session_type": "test"}
        )
        
        assert trail_id is not None
        assert len(trail_id) > 0
        assert trail_id in audit_system.active_trails
        
        # Log events to trail
        with audit_system.audit_context(sample_context):
            event_id = audit_system.log_event(
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                description="Test trade in trail"
            )
            
            # Add event to trail
            success = audit_system.log_to_trail(trail_id, event_id)
            assert success is True
        
        # End trail
        success = audit_system.end_trail(trail_id)
        assert success is True
        assert trail_id not in audit_system.active_trails
    
    def test_audit_event_handlers(self, audit_system, event_bus):
        """Test automatic audit event handlers"""
        # Publish trade event
        trade_event = event_bus.create_event(
            event_type=EventType.EXECUTE_TRADE,
            payload={
                "symbol": "ETH-USD",
                "quantity": 100,
                "price": 2000.0,
                "user_id": "test_user"
            },
            source="test"
        )
        
        initial_count = audit_system.event_count
        event_bus.publish(trade_event)
        
        # Should have automatically logged the event
        assert audit_system.event_count > initial_count
        
        # Check that event was logged
        events = audit_system.storage.get_events(limit=1)
        assert len(events) > 0
        assert events[0].event_type == AuditEventType.TRADE_EXECUTED
    
    def test_audit_summary_generation(self, audit_system, sample_context):
        """Test audit summary generation"""
        # Generate some events
        for i in range(5):
            audit_system.log_event(
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=sample_context,
                description=f"Test trade {i+1}"
            )
        
        # Generate summary
        summary = audit_system.get_audit_summary()
        
        assert "total_events" in summary
        assert "events_by_type" in summary
        assert "events_by_severity" in summary
        assert "events_by_outcome" in summary
        assert "unique_users" in summary
        assert "unique_components" in summary
        assert summary["total_events"] >= 5
    
    def test_audit_integrity_verification(self, audit_system, sample_context):
        """Test audit integrity verification"""
        # Generate some events
        for i in range(3):
            audit_system.log_event(
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=sample_context,
                description=f"Test trade {i+1}"
            )
        
        # Verify integrity
        integrity_report = audit_system.verify_audit_integrity()
        
        assert "total_events" in integrity_report
        assert "integrity_failures" in integrity_report
        assert "integrity_score" in integrity_report
        assert "is_tampered" in integrity_report
        assert integrity_report["integrity_score"] >= 0.0
        assert integrity_report["integrity_score"] <= 1.0
    
    def test_audit_system_status(self, audit_system):
        """Test audit system status reporting"""
        status = audit_system.get_system_status()
        
        assert "total_events" in status
        assert "total_trails" in status
        assert "active_trails" in status
        assert "storage_errors" in status
        assert "integrity_failures" in status
        assert "storage_path" in status
        assert status["total_events"] >= 0
        assert status["total_trails"] >= 0


class TestRegulatoryReporter:
    """Test the regulatory reporting system"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def compliance_monitor(self, event_bus):
        """Create compliance monitor for testing"""
        policy_engine = PolicyEngine(event_bus)
        return ComplianceMonitor(event_bus, policy_engine)
    
    @pytest.fixture
    def audit_system(self, event_bus):
        """Create audit system for testing"""
        return AuditSystem(event_bus)
    
    @pytest.fixture
    def regulatory_reporter(self, event_bus, compliance_monitor, audit_system, temp_output_dir):
        """Create regulatory reporter for testing"""
        return RegulatoryReporter(event_bus, compliance_monitor, audit_system, temp_output_dir)
    
    def test_report_template_creation(self):
        """Test report template creation"""
        template = TradeReportingTemplate.create_template()
        
        assert template.template_id == "trade_reporting_finra"
        assert template.report_type == ReportType.TRADE_REPORTING
        assert template.framework == RegulatoryFramework.FINRA
        assert template.format == ReportFormat.CSV
        assert template.frequency == ReportFrequency.DAILY
        assert len(template.fields) > 0
        assert len(template.required_fields) > 0
        assert len(template.field_mappings) > 0
    
    def test_report_validator(self):
        """Test report data validation"""
        template = TradeReportingTemplate.create_template()
        validator = ReportValidator(template)
        
        # Test valid data
        valid_data = [{
            "trade_id": "TRD001",
            "symbol": "ETH-USD",
            "trade_date": "2023-12-01",
            "trade_time": "09:30:00",
            "quantity": 100,
            "price": 2000.0,
            "side": "BUY",
            "venue": "EXCHANGE"
        }]
        
        is_valid, errors = validator.validate_data(valid_data)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid data (missing required field)
        invalid_data = [{
            "trade_id": "TRD001",
            "symbol": "ETH-USD",
            # Missing trade_date
            "quantity": 100,
            "price": 2000.0,
            "side": "BUY",
            "venue": "EXCHANGE"
        }]
        
        is_valid, errors = validator.validate_data(invalid_data)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_report_generator(self, temp_output_dir):
        """Test report generation"""
        template = TradeReportingTemplate.create_template()
        generator = ReportGenerator(template, temp_output_dir)
        
        # Test data
        data = [{
            "trade_id": "TRD001",
            "symbol": "ETH-USD",
            "trade_date": "2023-12-01",
            "trade_time": "09:30:00",
            "quantity": 100,
            "price": 2000.0,
            "side": "BUY",
            "venue": "EXCHANGE"
        }]
        
        # Generate report
        report = generator.generate_report(
            data=data,
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            metadata={"test": "data"}
        )
        
        assert report is not None
        assert report.template_id == template.template_id
        assert report.validation_status == "VALID"
        assert report.file_path is not None
        assert Path(report.file_path).exists()
        assert report.file_size > 0
        assert report.checksum is not None
    
    def test_csv_report_generation(self, temp_output_dir):
        """Test CSV report generation"""
        template = TradeReportingTemplate.create_template()
        generator = ReportGenerator(template, temp_output_dir)
        
        data = pd.DataFrame([{
            "trade_id": "TRD001",
            "symbol": "ETH-USD",
            "trade_date": "2023-12-01",
            "trade_time": "09:30:00",
            "quantity": 100,
            "price": 2000.0,
            "side": "BUY",
            "venue": "EXCHANGE"
        }])
        
        file_path, file_size, checksum = generator._generate_output(
            data, "test_report", datetime.now() - timedelta(days=1), datetime.now()
        )
        
        assert Path(file_path).exists()
        assert file_path.endswith('.csv')
        assert file_size > 0
        assert checksum is not None
        
        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == 1
        assert "TRADE_ID" in df.columns  # Should be mapped field name
    
    def test_xml_report_generation(self, temp_output_dir):
        """Test XML report generation"""
        template = PositionReportingTemplate.create_template()
        generator = ReportGenerator(template, temp_output_dir)
        
        data = pd.DataFrame([{
            "position_id": "POS001",
            "symbol": "ETH-USD",
            "position_date": "2023-12-01",
            "quantity": 100,
            "market_value": 200000.0,
            "asset_class": "EQUITY"
        }])
        
        file_path, file_size, checksum = generator._generate_output(
            data, "test_report", datetime.now() - timedelta(days=1), datetime.now()
        )
        
        assert Path(file_path).exists()
        assert file_path.endswith('.xml')
        assert file_size > 0
    
    def test_regulatory_reporter_initialization(self, regulatory_reporter):
        """Test regulatory reporter initialization"""
        assert regulatory_reporter is not None
        assert len(regulatory_reporter.templates) > 0
        assert len(regulatory_reporter.generators) > 0
        assert regulatory_reporter.output_dir.exists()
    
    def test_report_generation_integration(self, regulatory_reporter):
        """Test integrated report generation"""
        # Generate trade report
        report = regulatory_reporter.generate_report(
            template_id="trade_reporting_finra",
            data_source="trades",
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            metadata={"test": "integration"}
        )
        
        assert report is not None
        assert report.template_id == "trade_reporting_finra"
        assert report.report_type == ReportType.TRADE_REPORTING
        assert report.framework == RegulatoryFramework.FINRA
        assert report.file_path is not None
        assert Path(report.file_path).exists()
        
        # Check report was stored
        assert len(regulatory_reporter.reports) > 0
        assert report in regulatory_reporter.reports
    
    def test_report_status_tracking(self, regulatory_reporter):
        """Test report status tracking"""
        # Generate report
        report = regulatory_reporter.generate_report(
            template_id="trade_reporting_finra",
            data_source="trades"
        )
        
        if report:
            # Get report status
            status = regulatory_reporter.get_report_status(report.report_id)
            
            assert status is not None
            assert status["report_id"] == report.report_id
            assert status["validation_status"] == report.validation_status
            assert status["submission_status"] == report.submission_status
            assert status["file_path"] == report.file_path
    
    def test_report_filtering(self, regulatory_reporter):
        """Test report filtering and retrieval"""
        # Generate multiple reports
        for template_id in ["trade_reporting_finra", "position_reporting_sec"]:
            regulatory_reporter.generate_report(
                template_id=template_id,
                data_source="trades" if "trade" in template_id else "positions"
            )
        
        # Get all reports
        all_reports = regulatory_reporter.get_all_reports()
        assert len(all_reports) > 0
        
        # Filter by framework
        finra_reports = regulatory_reporter.get_all_reports(
            framework=RegulatoryFramework.FINRA
        )
        assert len(finra_reports) > 0
        assert all(r.framework == RegulatoryFramework.FINRA for r in finra_reports)
        
        # Filter by report type
        trade_reports = regulatory_reporter.get_all_reports(
            report_type=ReportType.TRADE_REPORTING
        )
        assert len(trade_reports) > 0
        assert all(r.report_type == ReportType.TRADE_REPORTING for r in trade_reports)
    
    def test_reporting_summary(self, regulatory_reporter):
        """Test reporting summary generation"""
        # Generate some reports
        for i in range(3):
            regulatory_reporter.generate_report(
                template_id="trade_reporting_finra",
                data_source="trades"
            )
        
        # Get summary
        summary = regulatory_reporter.get_reporting_summary()
        
        assert "total_reports" in summary
        assert "active_templates" in summary
        assert "reports_by_framework" in summary
        assert "reports_by_validation" in summary
        assert "reports_by_submission" in summary
        assert "recent_reports" in summary
        assert summary["total_reports"] >= 3
        assert summary["active_templates"] > 0
    
    def test_data_source_integration(self, regulatory_reporter):
        """Test different data source integrations"""
        # Test trade data
        trade_data = regulatory_reporter._get_trade_data(
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        assert isinstance(trade_data, pd.DataFrame)
        assert len(trade_data) > 0
        
        # Test position data
        position_data = regulatory_reporter._get_position_data(
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        assert isinstance(position_data, pd.DataFrame)
        assert len(position_data) > 0
        
        # Test risk data
        risk_data = regulatory_reporter._get_risk_data(
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        assert isinstance(risk_data, pd.DataFrame)
        assert len(risk_data) > 0


class TestGovernanceIntegration:
    """Test integrated governance system functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def governance_system(self, event_bus, temp_dir):
        """Create integrated governance system"""
        policy_engine = PolicyEngine(event_bus)
        compliance_monitor = ComplianceMonitor(
            event_bus, policy_engine, 
            str(Path(temp_dir) / "compliance.db")
        )
        audit_system = AuditSystem(
            event_bus, 
            AuditStorage(str(Path(temp_dir) / "audit.db"))
        )
        regulatory_reporter = RegulatoryReporter(
            event_bus, compliance_monitor, audit_system,
            str(Path(temp_dir) / "reports")
        )
        
        return {
            "policy_engine": policy_engine,
            "compliance_monitor": compliance_monitor,
            "audit_system": audit_system,
            "regulatory_reporter": regulatory_reporter
        }
    
    def test_policy_violation_audit_trail(self, governance_system, event_bus):
        """Test that policy violations create audit trails"""
        policy_engine = governance_system["policy_engine"]
        audit_system = governance_system["audit_system"]
        
        # Create context that will trigger violations
        context = PolicyContext(
            timestamp=datetime.now(),
            user_id="test_user",
            system_component="trading_engine",
            action_type="execute_trade",
            parameters={
                "symbol": "ETH-USD",
                "position_size": 50000.0,  # Large position
                "price": 2000.0
            },
            current_state={
                "portfolio_risk": {"var_95": 0.08}  # High risk
            }
        )
        
        initial_audit_count = audit_system.event_count
        
        # Evaluate policies (should trigger violations)
        violations = policy_engine.evaluate_policies(context)
        
        # Should have created audit events
        assert len(violations) > 0
        assert audit_system.event_count > initial_audit_count
        
        # Check audit events were created
        audit_events = audit_system.storage.get_events(limit=10)
        policy_events = [e for e in audit_events if "policy" in e.description.lower()]
        assert len(policy_events) > 0
    
    def test_compliance_violation_reporting(self, governance_system):
        """Test compliance violation reporting workflow"""
        compliance_monitor = governance_system["compliance_monitor"]
        regulatory_reporter = governance_system["regulatory_reporter"]
        
        # Create compliance violations
        context = {
            "risk_metrics": {
                "portfolio_var": 0.08,
                "leverage_ratio": 15.0
            },
            "trading_patterns": {
                "trades_per_hour": 1500,
                "order_cancel_ratio": 0.95
            }
        }
        
        violations = compliance_monitor.check_compliance(context)
        assert len(violations) > 0
        
        # Generate compliance report
        report = regulatory_reporter.generate_report(
            template_id="trade_reporting_finra",
            data_source="compliance",
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now()
        )
        
        assert report is not None
        assert report.validation_status in ["VALID", "INVALID"]
        assert report.file_path is not None
    
    def test_end_to_end_governance_workflow(self, governance_system, event_bus):
        """Test end-to-end governance workflow"""
        policy_engine = governance_system["policy_engine"]
        compliance_monitor = governance_system["compliance_monitor"]
        audit_system = governance_system["audit_system"]
        regulatory_reporter = governance_system["regulatory_reporter"]
        
        # Start audit trail
        trail_id = audit_system.start_trail("Test Trading Session")
        
        # Create trading context
        context = PolicyContext(
            timestamp=datetime.now(),
            user_id="trader001",
            system_component="trading_engine",
            action_type="execute_trade",
            parameters={
                "symbol": "ETH-USD",
                "position_size": 15000.0,
                "price": 2000.0
            },
            current_state={
                "portfolio_risk": {"var_95": 0.06}
            },
            portfolio_data={
                "total_value": 1000000.0,
                "current_drawdown": 0.12
            }
        )
        
        # 1. Policy evaluation
        policy_violations = policy_engine.evaluate_policies(context)
        assert len(policy_violations) > 0
        
        # 2. Compliance checking
        compliance_context = {
            "risk_metrics": {
                "portfolio_var": 0.06,
                "leverage_ratio": 8.0
            }
        }
        
        compliance_violations = compliance_monitor.check_compliance(compliance_context)
        assert len(compliance_violations) > 0
        
        # 3. Audit logging
        audit_context = AuditContext(
            user_id="trader001",
            session_id="session001",
            ip_address="192.168.1.100",
            user_agent="trading_client",
            system_component="trading_engine",
            action="execute_trade",
            resource="ETH-USD"
        )
        
        event_id = audit_system.log_event(
            event_type=AuditEventType.TRADE_EXECUTED,
            severity=AuditSeverity.WARNING,
            outcome=AuditOutcome.PARTIAL,
            context=audit_context,
            description="Trade executed with policy violations"
        )
        
        # Add to trail
        audit_system.log_to_trail(trail_id, event_id)
        
        # 4. Generate reports
        reports = []
        for template_id in ["trade_reporting_finra", "position_reporting_sec"]:
            report = regulatory_reporter.generate_report(
                template_id=template_id,
                data_source="trades" if "trade" in template_id else "positions"
            )
            if report:
                reports.append(report)
        
        # 5. End audit trail
        audit_system.end_trail(trail_id)
        
        # Verify complete workflow
        assert len(reports) > 0
        assert audit_system.event_count > 0
        assert compliance_monitor.violation_count > 0
        assert policy_engine.violation_count > 0
        
        # Generate summary reports
        audit_summary = audit_system.get_audit_summary()
        compliance_status = compliance_monitor.get_compliance_status()
        reporting_summary = regulatory_reporter.get_reporting_summary()
        
        assert audit_summary["total_events"] > 0
        assert compliance_status["total_violations"] > 0
        assert reporting_summary["total_reports"] > 0
    
    def test_system_performance_under_load(self, governance_system):
        """Test system performance under load"""
        import time
        
        policy_engine = governance_system["policy_engine"]
        compliance_monitor = governance_system["compliance_monitor"]
        audit_system = governance_system["audit_system"]
        
        # Generate load
        start_time = time.time()
        
        for i in range(100):
            # Policy evaluation
            context = PolicyContext(
                timestamp=datetime.now(),
                user_id=f"user_{i}",
                system_component="trading_engine",
                action_type="execute_trade",
                parameters={
                    "symbol": "ETH-USD",
                    "position_size": 1000.0 + i,
                    "price": 2000.0
                },
                current_state={
                    "portfolio_risk": {"var_95": 0.02 + (i * 0.0001)}
                }
            )
            
            policy_engine.evaluate_policies(context)
            
            # Compliance checking
            compliance_context = {
                "risk_metrics": {
                    "portfolio_var": 0.02 + (i * 0.0001),
                    "leverage_ratio": 2.0 + (i * 0.01)
                }
            }
            
            compliance_monitor.check_compliance(compliance_context)
            
            # Audit logging
            audit_context = AuditContext(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                ip_address="192.168.1.100",
                user_agent="test_client",
                system_component="test_system",
                action="test_action",
                resource="test_resource"
            )
            
            audit_system.log_event(
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=audit_context,
                description=f"Test event {i}"
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 30.0  # Should complete in under 30 seconds
        assert policy_engine.evaluation_count >= 100
        assert compliance_monitor.total_checks >= 100
        assert audit_system.event_count >= 100
        
        # Verify system still functions correctly
        assert policy_engine.enabled is True
        assert compliance_monitor.compliance_score >= 0.0
        assert audit_system.storage_errors == 0


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])