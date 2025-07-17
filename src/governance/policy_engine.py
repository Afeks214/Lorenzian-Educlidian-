"""
Policy Engine for Trading System Governance

This module implements policy enforcement, rule evaluation, and violation detection
for comprehensive trading system governance.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import structlog
from abc import ABC, abstractmethod

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class PolicyType(Enum):
    """Types of governance policies"""
    POSITION_LIMIT = "position_limit"
    RISK_LIMIT = "risk_limit"
    TRADING_HOURS = "trading_hours"
    CONCENTRATION_LIMIT = "concentration_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    COMPLIANCE_RULE = "compliance_rule"
    OPERATIONAL_RULE = "operational_rule"


class PolicySeverity(Enum):
    """Severity levels for policy violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(Enum):
    """Actions to take when policy is violated"""
    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"
    TERMINATE = "terminate"
    ESCALATE = "escalate"


@dataclass
class PolicyContext:
    """Context information for policy evaluation"""
    timestamp: datetime
    user_id: str
    system_component: str
    action_type: str
    parameters: Dict[str, Any]
    current_state: Dict[str, Any]
    portfolio_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None


@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    policy_id: str
    policy_name: str
    severity: PolicySeverity
    message: str
    context: PolicyContext
    timestamp: datetime
    violation_value: Any
    threshold_value: Any
    suggested_action: PolicyAction
    violation_id: str = field(default_factory=lambda: f"violation_{int(time.time() * 1000)}")
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class PolicyRule(ABC):
    """Abstract base class for policy rules"""
    
    def __init__(
        self,
        policy_id: str,
        policy_name: str,
        policy_type: PolicyType,
        severity: PolicySeverity,
        action: PolicyAction,
        enabled: bool = True,
        description: str = ""
    ):
        self.policy_id = policy_id
        self.policy_name = policy_name
        self.policy_type = policy_type
        self.severity = severity
        self.action = action
        self.enabled = enabled
        self.description = description
        self.violation_count = 0
        self.last_violation = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @abstractmethod
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate the policy rule against given context"""
        pass
    
    def is_applicable(self, context: PolicyContext) -> bool:
        """Check if policy rule is applicable to given context"""
        return self.enabled
    
    def update_violation_stats(self, violation: PolicyViolation):
        """Update violation statistics"""
        self.violation_count += 1
        self.last_violation = violation
        self.updated_at = datetime.now()


class PositionLimitPolicy(PolicyRule):
    """Policy for position size limits"""
    
    def __init__(
        self,
        policy_id: str,
        max_position_size: float,
        symbol: Optional[str] = None,
        severity: PolicySeverity = PolicySeverity.HIGH,
        action: PolicyAction = PolicyAction.BLOCK
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name=f"Position Limit - {symbol or 'All Symbols'}",
            policy_type=PolicyType.POSITION_LIMIT,
            severity=severity,
            action=action,
            description=f"Maximum position size: {max_position_size}"
        )
        self.max_position_size = max_position_size
        self.symbol = symbol
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate position size limit"""
        if not self.is_applicable(context):
            return None
        
        # Extract position information
        position_size = context.parameters.get("position_size", 0)
        symbol = context.parameters.get("symbol", "")
        
        # Check if policy applies to this symbol
        if self.symbol and self.symbol != symbol:
            return None
        
        # Check if position size exceeds limit
        if abs(position_size) > self.max_position_size:
            violation = PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.policy_name,
                severity=self.severity,
                message=f"Position size {position_size} exceeds limit of {self.max_position_size} for {symbol}",
                context=context,
                timestamp=datetime.now(),
                violation_value=abs(position_size),
                threshold_value=self.max_position_size,
                suggested_action=self.action
            )
            
            self.update_violation_stats(violation)
            return violation
        
        return None


class RiskLimitPolicy(PolicyRule):
    """Policy for risk limits"""
    
    def __init__(
        self,
        policy_id: str,
        max_portfolio_risk: float,
        risk_measure: str = "var_95",
        severity: PolicySeverity = PolicySeverity.CRITICAL,
        action: PolicyAction = PolicyAction.REDUCE
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name=f"Risk Limit - {risk_measure}",
            policy_type=PolicyType.RISK_LIMIT,
            severity=severity,
            action=action,
            description=f"Maximum portfolio risk ({risk_measure}): {max_portfolio_risk}"
        )
        self.max_portfolio_risk = max_portfolio_risk
        self.risk_measure = risk_measure
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate risk limit"""
        if not self.is_applicable(context):
            return None
        
        # Extract risk information
        portfolio_risk = context.current_state.get("portfolio_risk", {})
        current_risk = portfolio_risk.get(self.risk_measure, 0)
        
        # Check if risk exceeds limit
        if current_risk > self.max_portfolio_risk:
            violation = PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.policy_name,
                severity=self.severity,
                message=f"Portfolio risk ({self.risk_measure}) {current_risk:.4f} exceeds limit of {self.max_portfolio_risk:.4f}",
                context=context,
                timestamp=datetime.now(),
                violation_value=current_risk,
                threshold_value=self.max_portfolio_risk,
                suggested_action=self.action
            )
            
            self.update_violation_stats(violation)
            return violation
        
        return None


class TradingHoursPolicy(PolicyRule):
    """Policy for trading hours restrictions"""
    
    def __init__(
        self,
        policy_id: str,
        allowed_hours: List[int],
        timezone: str = "UTC",
        severity: PolicySeverity = PolicySeverity.MEDIUM,
        action: PolicyAction = PolicyAction.BLOCK
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name="Trading Hours Restriction",
            policy_type=PolicyType.TRADING_HOURS,
            severity=severity,
            action=action,
            description=f"Allowed trading hours: {allowed_hours} ({timezone})"
        )
        self.allowed_hours = allowed_hours
        self.timezone = timezone
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate trading hours restriction"""
        if not self.is_applicable(context):
            return None
        
        # Check if this is a trading action
        if context.action_type not in ["place_order", "execute_trade", "modify_position"]:
            return None
        
        # Get current hour
        current_hour = context.timestamp.hour
        
        # Check if current hour is allowed
        if current_hour not in self.allowed_hours:
            violation = PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.policy_name,
                severity=self.severity,
                message=f"Trading not allowed at hour {current_hour}. Allowed hours: {self.allowed_hours}",
                context=context,
                timestamp=datetime.now(),
                violation_value=current_hour,
                threshold_value=self.allowed_hours,
                suggested_action=self.action
            )
            
            self.update_violation_stats(violation)
            return violation
        
        return None


class ConcentrationLimitPolicy(PolicyRule):
    """Policy for concentration limits"""
    
    def __init__(
        self,
        policy_id: str,
        max_concentration: float,
        concentration_type: str = "symbol",
        severity: PolicySeverity = PolicySeverity.HIGH,
        action: PolicyAction = PolicyAction.WARN
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name=f"Concentration Limit - {concentration_type}",
            policy_type=PolicyType.CONCENTRATION_LIMIT,
            severity=severity,
            action=action,
            description=f"Maximum concentration ({concentration_type}): {max_concentration:.2%}"
        )
        self.max_concentration = max_concentration
        self.concentration_type = concentration_type
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate concentration limit"""
        if not self.is_applicable(context):
            return None
        
        # Extract portfolio data
        portfolio_data = context.portfolio_data
        if not portfolio_data:
            return None
        
        # Calculate concentration
        concentration = self._calculate_concentration(portfolio_data, context.parameters)
        
        # Check if concentration exceeds limit
        if concentration > self.max_concentration:
            violation = PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.policy_name,
                severity=self.severity,
                message=f"Concentration ({self.concentration_type}) {concentration:.2%} exceeds limit of {self.max_concentration:.2%}",
                context=context,
                timestamp=datetime.now(),
                violation_value=concentration,
                threshold_value=self.max_concentration,
                suggested_action=self.action
            )
            
            self.update_violation_stats(violation)
            return violation
        
        return None
    
    def _calculate_concentration(self, portfolio_data: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """Calculate concentration based on type"""
        if self.concentration_type == "symbol":
            symbol = parameters.get("symbol", "")
            total_value = portfolio_data.get("total_value", 1.0)
            symbol_value = portfolio_data.get("positions", {}).get(symbol, {}).get("value", 0.0)
            return abs(symbol_value) / total_value if total_value > 0 else 0.0
        
        elif self.concentration_type == "sector":
            # Mock sector concentration calculation
            return 0.15  # 15% concentration
        
        return 0.0


class DrawdownLimitPolicy(PolicyRule):
    """Policy for drawdown limits"""
    
    def __init__(
        self,
        policy_id: str,
        max_drawdown: float,
        lookback_period: int = 30,
        severity: PolicySeverity = PolicySeverity.CRITICAL,
        action: PolicyAction = PolicyAction.TERMINATE
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name="Drawdown Limit",
            policy_type=PolicyType.DRAWDOWN_LIMIT,
            severity=severity,
            action=action,
            description=f"Maximum drawdown: {max_drawdown:.2%} over {lookback_period} days"
        )
        self.max_drawdown = max_drawdown
        self.lookback_period = lookback_period
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate drawdown limit"""
        if not self.is_applicable(context):
            return None
        
        # Extract portfolio performance data
        portfolio_data = context.portfolio_data
        if not portfolio_data:
            return None
        
        current_drawdown = portfolio_data.get("current_drawdown", 0.0)
        
        # Check if drawdown exceeds limit
        if current_drawdown > self.max_drawdown:
            violation = PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.policy_name,
                severity=self.severity,
                message=f"Current drawdown {current_drawdown:.2%} exceeds limit of {self.max_drawdown:.2%}",
                context=context,
                timestamp=datetime.now(),
                violation_value=current_drawdown,
                threshold_value=self.max_drawdown,
                suggested_action=self.action
            )
            
            self.update_violation_stats(violation)
            return violation
        
        return None


class CustomFunctionPolicy(PolicyRule):
    """Policy that uses a custom function for evaluation"""
    
    def __init__(
        self,
        policy_id: str,
        policy_name: str,
        evaluation_function: Callable[[PolicyContext], Optional[PolicyViolation]],
        policy_type: PolicyType = PolicyType.COMPLIANCE_RULE,
        severity: PolicySeverity = PolicySeverity.MEDIUM,
        action: PolicyAction = PolicyAction.WARN
    ):
        super().__init__(
            policy_id=policy_id,
            policy_name=policy_name,
            policy_type=policy_type,
            severity=severity,
            action=action,
            description="Custom policy evaluation function"
        )
        self.evaluation_function = evaluation_function
    
    def evaluate(self, context: PolicyContext) -> Optional[PolicyViolation]:
        """Evaluate using custom function"""
        if not self.is_applicable(context):
            return None
        
        try:
            violation = self.evaluation_function(context)
            if violation:
                self.update_violation_stats(violation)
            return violation
        except Exception as e:
            logger.error("Error in custom policy evaluation", policy_id=self.policy_id, error=str(e))
            return None


class PolicyEngine:
    """Main policy enforcement engine"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.policies: Dict[str, PolicyRule] = {}
        self.violations: List[PolicyViolation] = []
        self.policy_stats: Dict[str, Dict[str, Any]] = {}
        self.enabled = True
        
        # Performance tracking
        self.evaluation_count = 0
        self.violation_count = 0
        self.last_evaluation_time = None
        
        # Event subscriptions
        self._setup_event_handlers()
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Policy Engine initialized")
    
    def _setup_event_handlers(self):
        """Setup event handlers for policy evaluation triggers"""
        self.event_bus.subscribe(EventType.EXECUTE_TRADE, self._handle_trade_event)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_event)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_event)
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_strategic_event)
    
    def _initialize_default_policies(self):
        """Initialize default policies"""
        # Position limit policies
        self.add_policy(PositionLimitPolicy(
            policy_id="pos_limit_eth",
            max_position_size=10000.0,
            symbol="ETH-USD",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.BLOCK
        ))
        
        self.add_policy(PositionLimitPolicy(
            policy_id="pos_limit_btc",
            max_position_size=1.0,
            symbol="BTC-USD",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.BLOCK
        ))
        
        # Risk limit policy
        self.add_policy(RiskLimitPolicy(
            policy_id="portfolio_var_limit",
            max_portfolio_risk=0.05,
            risk_measure="var_95",
            severity=PolicySeverity.CRITICAL,
            action=PolicyAction.REDUCE
        ))
        
        # Trading hours policy
        self.add_policy(TradingHoursPolicy(
            policy_id="trading_hours",
            allowed_hours=list(range(9, 17)),  # 9 AM to 5 PM
            timezone="UTC",
            severity=PolicySeverity.MEDIUM,
            action=PolicyAction.BLOCK
        ))
        
        # Concentration limit policy
        self.add_policy(ConcentrationLimitPolicy(
            policy_id="symbol_concentration",
            max_concentration=0.25,  # 25% max concentration
            concentration_type="symbol",
            severity=PolicySeverity.HIGH,
            action=PolicyAction.WARN
        ))
        
        # Drawdown limit policy
        self.add_policy(DrawdownLimitPolicy(
            policy_id="max_drawdown",
            max_drawdown=0.20,  # 20% max drawdown
            lookback_period=30,
            severity=PolicySeverity.CRITICAL,
            action=PolicyAction.TERMINATE
        ))
    
    def add_policy(self, policy: PolicyRule) -> bool:
        """Add a policy to the engine"""
        try:
            self.policies[policy.policy_id] = policy
            self.policy_stats[policy.policy_id] = {
                "evaluations": 0,
                "violations": 0,
                "last_evaluation": None,
                "last_violation": None
            }
            
            logger.info("Policy added", policy_id=policy.policy_id, policy_name=policy.policy_name)
            return True
            
        except Exception as e:
            logger.error("Failed to add policy", policy_id=policy.policy_id, error=str(e))
            return False
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the engine"""
        try:
            if policy_id in self.policies:
                del self.policies[policy_id]
                if policy_id in self.policy_stats:
                    del self.policy_stats[policy_id]
                
                logger.info("Policy removed", policy_id=policy_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to remove policy", policy_id=policy_id, error=str(e))
            return False
    
    def enable_policy(self, policy_id: str) -> bool:
        """Enable a policy"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            logger.info("Policy enabled", policy_id=policy_id)
            return True
        return False
    
    def disable_policy(self, policy_id: str) -> bool:
        """Disable a policy"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            logger.info("Policy disabled", policy_id=policy_id)
            return True
        return False
    
    def evaluate_policies(self, context: PolicyContext) -> List[PolicyViolation]:
        """Evaluate all applicable policies against given context"""
        if not self.enabled:
            return []
        
        violations = []
        self.evaluation_count += 1
        self.last_evaluation_time = datetime.now()
        
        for policy_id, policy in self.policies.items():
            try:
                # Update policy stats
                self.policy_stats[policy_id]["evaluations"] += 1
                self.policy_stats[policy_id]["last_evaluation"] = datetime.now()
                
                # Evaluate policy
                violation = policy.evaluate(context)
                
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)
                    self.violation_count += 1
                    
                    # Update policy stats
                    self.policy_stats[policy_id]["violations"] += 1
                    self.policy_stats[policy_id]["last_violation"] = datetime.now()
                    
                    # Publish violation event
                    self._publish_violation_event(violation)
                    
                    logger.warning(
                        "Policy violation detected",
                        policy_id=policy_id,
                        violation_id=violation.violation_id,
                        severity=violation.severity.value,
                        message=violation.message
                    )
                
            except Exception as e:
                logger.error("Error evaluating policy", policy_id=policy_id, error=str(e))
                continue
        
        return violations
    
    def _publish_violation_event(self, violation: PolicyViolation):
        """Publish policy violation event"""
        event = self.event_bus.create_event(
            event_type=EventType.RISK_BREACH,  # Using existing event type
            payload={
                "type": "policy_violation",
                "violation_id": violation.violation_id,
                "policy_id": violation.policy_id,
                "severity": violation.severity.value,
                "message": violation.message,
                "suggested_action": violation.suggested_action.value,
                "timestamp": violation.timestamp.isoformat()
            },
            source="policy_engine"
        )
        
        self.event_bus.publish(event)
    
    def _handle_trade_event(self, event: Event):
        """Handle trade execution events"""
        try:
            payload = event.payload
            
            context = PolicyContext(
                timestamp=datetime.now(),
                user_id=payload.get("user_id", "system"),
                system_component="trading_engine",
                action_type="execute_trade",
                parameters={
                    "symbol": payload.get("symbol", ""),
                    "position_size": payload.get("position_size", 0),
                    "price": payload.get("price", 0),
                    "order_type": payload.get("order_type", "")
                },
                current_state=payload.get("current_state", {}),
                portfolio_data=payload.get("portfolio_data"),
                market_data=payload.get("market_data")
            )
            
            violations = self.evaluate_policies(context)
            
            # Take action on violations
            for violation in violations:
                self._handle_violation_action(violation)
                
        except Exception as e:
            logger.error("Error handling trade event", error=str(e))
    
    def _handle_position_event(self, event: Event):
        """Handle position update events"""
        try:
            payload = event.payload
            
            context = PolicyContext(
                timestamp=datetime.now(),
                user_id=payload.get("user_id", "system"),
                system_component="position_manager",
                action_type="update_position",
                parameters={
                    "symbol": payload.get("symbol", ""),
                    "position_size": payload.get("position_size", 0),
                    "position_value": payload.get("position_value", 0)
                },
                current_state=payload.get("current_state", {}),
                portfolio_data=payload.get("portfolio_data"),
                market_data=payload.get("market_data")
            )
            
            violations = self.evaluate_policies(context)
            
            # Take action on violations
            for violation in violations:
                self._handle_violation_action(violation)
                
        except Exception as e:
            logger.error("Error handling position event", error=str(e))
    
    def _handle_risk_event(self, event: Event):
        """Handle risk breach events"""
        try:
            payload = event.payload
            
            context = PolicyContext(
                timestamp=datetime.now(),
                user_id=payload.get("user_id", "system"),
                system_component="risk_engine",
                action_type="risk_breach",
                parameters=payload.get("parameters", {}),
                current_state=payload.get("current_state", {}),
                portfolio_data=payload.get("portfolio_data"),
                market_data=payload.get("market_data")
            )
            
            violations = self.evaluate_policies(context)
            
            # Take action on violations
            for violation in violations:
                self._handle_violation_action(violation)
                
        except Exception as e:
            logger.error("Error handling risk event", error=str(e))
    
    def _handle_strategic_event(self, event: Event):
        """Handle strategic decision events"""
        try:
            payload = event.payload
            
            context = PolicyContext(
                timestamp=datetime.now(),
                user_id=payload.get("user_id", "system"),
                system_component="strategic_engine",
                action_type="strategic_decision",
                parameters=payload.get("parameters", {}),
                current_state=payload.get("current_state", {}),
                portfolio_data=payload.get("portfolio_data"),
                market_data=payload.get("market_data")
            )
            
            violations = self.evaluate_policies(context)
            
            # Take action on violations
            for violation in violations:
                self._handle_violation_action(violation)
                
        except Exception as e:
            logger.error("Error handling strategic event", error=str(e))
    
    def _handle_violation_action(self, violation: PolicyViolation):
        """Handle violation action based on policy"""
        try:
            if violation.suggested_action == PolicyAction.WARN:
                logger.warning("Policy violation warning", violation_id=violation.violation_id)
            
            elif violation.suggested_action == PolicyAction.BLOCK:
                logger.error("Policy violation blocked action", violation_id=violation.violation_id)
                # Could implement actual blocking logic here
            
            elif violation.suggested_action == PolicyAction.REDUCE:
                logger.error("Policy violation requires position reduction", violation_id=violation.violation_id)
                # Could implement position reduction logic here
            
            elif violation.suggested_action == PolicyAction.TERMINATE:
                logger.critical("Policy violation requires termination", violation_id=violation.violation_id)
                # Could implement termination logic here
            
            elif violation.suggested_action == PolicyAction.ESCALATE:
                logger.critical("Policy violation escalated", violation_id=violation.violation_id)
                # Could implement escalation logic here
                
        except Exception as e:
            logger.error("Error handling violation action", violation_id=violation.violation_id, error=str(e))
    
    def get_policy_status(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific policy"""
        if policy_id not in self.policies:
            return None
        
        policy = self.policies[policy_id]
        stats = self.policy_stats[policy_id]
        
        return {
            "policy_id": policy.policy_id,
            "policy_name": policy.policy_name,
            "policy_type": policy.policy_type.value,
            "enabled": policy.enabled,
            "severity": policy.severity.value,
            "action": policy.action.value,
            "description": policy.description,
            "violation_count": policy.violation_count,
            "last_violation": policy.last_violation.timestamp.isoformat() if policy.last_violation else None,
            "stats": stats
        }
    
    def get_all_policies_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all policies"""
        return {
            policy_id: self.get_policy_status(policy_id)
            for policy_id in self.policies.keys()
        }
    
    def get_violations(
        self,
        policy_id: Optional[str] = None,
        severity: Optional[PolicySeverity] = None,
        resolved: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[PolicyViolation]:
        """Get violations with optional filtering"""
        violations = self.violations.copy()
        
        # Apply filters
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        
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
    
    def resolve_violation(self, violation_id: str, resolution_notes: str = "") -> bool:
        """Mark a violation as resolved"""
        try:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution_timestamp = datetime.now()
                    violation.resolution_notes = resolution_notes
                    
                    logger.info("Violation resolved", violation_id=violation_id)
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error resolving violation", violation_id=violation_id, error=str(e))
            return False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get overall engine status"""
        return {
            "enabled": self.enabled,
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.enabled]),
            "total_evaluations": self.evaluation_count,
            "total_violations": self.violation_count,
            "unresolved_violations": len([v for v in self.violations if not v.resolved]),
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "policy_stats": self.policy_stats
        }
    
    def enable_engine(self):
        """Enable the policy engine"""
        self.enabled = True
        logger.info("Policy engine enabled")
    
    def disable_engine(self):
        """Disable the policy engine"""
        self.enabled = False
        logger.info("Policy engine disabled")
    
    def cleanup_old_violations(self, days_old: int = 30):
        """Clean up old resolved violations"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        initial_count = len(self.violations)
        self.violations = [
            v for v in self.violations
            if not v.resolved or v.resolution_timestamp > cutoff_date
        ]
        
        cleaned_count = initial_count - len(self.violations)
        
        if cleaned_count > 0:
            logger.info("Cleaned up old violations", count=cleaned_count)
        
        return cleaned_count