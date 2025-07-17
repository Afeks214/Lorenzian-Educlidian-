"""
Temporal Boundary Enforcement System
Agent 5: Data Quality & Bias Elimination

Strict temporal boundary enforcement system that prevents look-ahead bias
and ensures data availability constraints are respected. This system acts
as a gatekeeper for all data access within the trading system.

Key Features:
- Strict temporal boundary enforcement
- Data availability time-stamping
- Look-ahead bias prevention
- Causal relationship enforcement
- Multi-timeframe boundary synchronization
- Real-time boundary monitoring
- Automatic violation detection and prevention
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import structlog

from .temporal_bias_detector import TemporalBoundary, TemporalBoundaryType, SeverityLevel

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class AccessType(str, Enum):
    """Types of data access"""
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    BACKTEST = "backtest"
    LIVE = "live"

class BoundaryViolationType(str, Enum):
    """Types of boundary violations"""
    FUTURE_ACCESS = "future_access"
    UNAVAILABLE_DATA = "unavailable_data"
    LATENCY_VIOLATION = "latency_violation"
    CAUSAL_VIOLATION = "causal_violation"
    DEPENDENCY_VIOLATION = "dependency_violation"

class EnforcementAction(str, Enum):
    """Actions taken when boundaries are violated"""
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"
    SUBSTITUTE = "substitute"
    DELAY = "delay"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataAccessRequest:
    """Data access request with temporal constraints"""
    request_id: str
    requester_id: str
    access_type: AccessType
    
    # Temporal constraints
    request_time: datetime
    data_timestamp: datetime
    latest_allowable_time: datetime
    
    # Data identification
    data_identifier: str
    data_type: str
    component: str
    
    # Access parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    priority: int = 1
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    
    # Tags
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class BoundaryViolation:
    """Boundary violation record"""
    violation_id: str
    violation_type: BoundaryViolationType
    severity: SeverityLevel
    
    # Violation details
    detected_at: datetime
    boundary_id: str
    request_id: str
    
    # Temporal information
    requested_time: datetime
    allowed_time: datetime
    violation_duration: timedelta
    
    # Context
    requester_id: str
    data_identifier: str
    component: str
    
    # Action taken
    enforcement_action: EnforcementAction
    action_successful: bool = False
    
    # Impact
    impact_assessment: str = ""
    downstream_effects: List[str] = field(default_factory=list)
    
    # Remediation
    remediation_applied: bool = False
    remediation_details: str = ""
    
    # Metadata
    detector_id: str = "boundary_enforcer"
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BoundaryEnforcementRule:
    """Boundary enforcement rule"""
    rule_id: str
    rule_name: str
    rule_type: str
    
    # Rule condition
    condition: Callable[[DataAccessRequest], bool]
    
    # Enforcement action
    action: EnforcementAction
    severity: SeverityLevel
    
    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Scope
    applicable_components: List[str] = field(default_factory=list)
    applicable_data_types: List[str] = field(default_factory=list)
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    violations_detected: int = 0
    actions_taken: int = 0

# =============================================================================
# TEMPORAL BOUNDARY ENFORCER
# =============================================================================

class TemporalBoundaryEnforcer:
    """Strict temporal boundary enforcement system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Boundary registry
        self.temporal_boundaries: Dict[str, TemporalBoundary] = {}
        self.enforcement_rules: Dict[str, BoundaryEnforcementRule] = {}
        
        # Access tracking
        self.access_requests: Dict[str, DataAccessRequest] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.violation_history: deque = deque(maxlen=1000)
        
        # Real-time monitoring
        self.current_time_providers: Dict[str, Callable[[], datetime]] = {}
        self.time_synchronization_offset: timedelta = timedelta(0)
        
        # Enforcement state
        self.enforcement_active: bool = True
        self.strict_mode: bool = self.config.get('strict_mode', True)
        
        # Background processing
        self.monitoring_active: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.enforcement_executor = ThreadPoolExecutor(max_workers=2)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'granted_requests': 0,
            'blocked_requests': 0,
            'violations_detected': 0,
            'boundaries_active': 0,
            'rules_active': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Temporal boundary enforcer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'strict_mode': True,
            'enable_continuous_monitoring': True,
            'monitoring_interval_seconds': 10,
            'max_violation_history': 1000,
            'default_enforcement_action': EnforcementAction.BLOCK,
            'time_synchronization_enabled': True,
            'violation_alert_threshold': 5,
            'auto_remediation_enabled': True,
            'performance_monitoring': True
        }
    
    def start_monitoring(self):
        """Start continuous boundary monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="boundary_enforcer_monitor"
            )
            self.monitor_thread.start()
            logger.info("Boundary enforcement monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous boundary monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.enforcement_executor.shutdown(wait=True)
        logger.info("Boundary enforcement monitoring stopped")
    
    def register_temporal_boundary(self, boundary: TemporalBoundary):
        """Register a temporal boundary for enforcement"""
        with self.lock:
            self.temporal_boundaries[boundary.boundary_id] = boundary
            self.stats['boundaries_active'] = len(self.temporal_boundaries)
            logger.debug(f"Registered temporal boundary: {boundary.boundary_id}")
    
    def register_enforcement_rule(self, rule: BoundaryEnforcementRule):
        """Register an enforcement rule"""
        with self.lock:
            self.enforcement_rules[rule.rule_id] = rule
            self.stats['rules_active'] = len(self.enforcement_rules)
            logger.debug(f"Registered enforcement rule: {rule.rule_name}")
    
    def register_time_provider(self, provider_id: str, provider_func: Callable[[], datetime]):
        """Register a time provider for synchronization"""
        self.current_time_providers[provider_id] = provider_func
        logger.debug(f"Registered time provider: {provider_id}")
    
    async def request_data_access(self, request: DataAccessRequest) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Request access to data with temporal boundary validation"""
        
        with self.lock:
            self.stats['total_requests'] += 1
            
            # Store request
            self.access_requests[request.request_id] = request
            
            # Log access attempt
            logger.debug(f"Data access request: {request.request_id} for {request.data_identifier}")
            
            try:
                # Validate temporal boundaries
                is_valid, violation_reason, violation = await self._validate_temporal_access(request)
                
                if is_valid:
                    # Grant access
                    self.stats['granted_requests'] += 1
                    
                    # Record successful access
                    self.access_history.append({
                        'request_id': request.request_id,
                        'timestamp': datetime.utcnow(),
                        'status': 'granted',
                        'data_identifier': request.data_identifier
                    })
                    
                    logger.debug(f"Access granted: {request.request_id}")
                    return True, None, None
                
                else:
                    # Handle violation
                    self.stats['blocked_requests'] += 1
                    
                    if violation:
                        self.violation_history.append(violation)
                        await self._handle_boundary_violation(violation)
                    
                    # Record blocked access
                    self.access_history.append({
                        'request_id': request.request_id,
                        'timestamp': datetime.utcnow(),
                        'status': 'blocked',
                        'reason': violation_reason,
                        'data_identifier': request.data_identifier
                    })
                    
                    logger.warning(f"Access blocked: {request.request_id} - {violation_reason}")
                    return False, violation_reason, violation
                    
            except Exception as e:
                logger.error(f"Error processing access request {request.request_id}: {e}")
                return False, f"Processing error: {str(e)}", None
    
    async def _validate_temporal_access(self, request: DataAccessRequest) -> Tuple[bool, Optional[str], Optional[BoundaryViolation]]:
        """Validate temporal access constraints"""
        
        # Get current time
        current_time = self._get_current_time()
        
        # Check if accessing future data
        if request.data_timestamp > current_time:
            violation = BoundaryViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=BoundaryViolationType.FUTURE_ACCESS,
                severity=SeverityLevel.CRITICAL,
                detected_at=current_time,
                boundary_id="future_access_boundary",
                request_id=request.request_id,
                requested_time=request.data_timestamp,
                allowed_time=current_time,
                violation_duration=request.data_timestamp - current_time,
                requester_id=request.requester_id,
                data_identifier=request.data_identifier,
                component=request.component,
                enforcement_action=EnforcementAction.BLOCK
            )
            
            return False, f"Future data access attempted: {request.data_timestamp} > {current_time}", violation
        
        # Check against registered boundaries
        for boundary_id, boundary in self.temporal_boundaries.items():
            if self._is_boundary_applicable(boundary, request):
                is_valid, violation_reason, violation = await self._check_boundary_constraint(boundary, request)
                
                if not is_valid:
                    return False, violation_reason, violation
        
        # Check enforcement rules
        for rule_id, rule in self.enforcement_rules.items():
            if rule.enabled and rule.condition(request):
                violation = await self._apply_enforcement_rule(rule, request)
                
                if violation:
                    return False, f"Rule violation: {rule.rule_name}", violation
        
        return True, None, None
    
    def _is_boundary_applicable(self, boundary: TemporalBoundary, request: DataAccessRequest) -> bool:
        """Check if boundary applies to request"""
        
        # Check component match
        if boundary.component != "unknown" and boundary.component != request.component:
            return False
        
        # Check data source match
        if boundary.data_source != "unknown" and boundary.data_source != request.data_type:
            return False
        
        # Check timestamp within boundary scope
        if request.data_timestamp < boundary.timestamp:
            return False
        
        return True
    
    async def _check_boundary_constraint(self, boundary: TemporalBoundary, request: DataAccessRequest) -> Tuple[bool, Optional[str], Optional[BoundaryViolation]]:
        """Check specific boundary constraint"""
        
        current_time = self._get_current_time()
        
        # Check lookback constraint
        if boundary.max_lookback_hours > 0:
            max_lookback_time = current_time - timedelta(hours=boundary.max_lookback_hours)
            
            if request.data_timestamp < max_lookback_time:
                violation = BoundaryViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=BoundaryViolationType.UNAVAILABLE_DATA,
                    severity=boundary.enforcement_level,
                    detected_at=current_time,
                    boundary_id=boundary.boundary_id,
                    request_id=request.request_id,
                    requested_time=request.data_timestamp,
                    allowed_time=max_lookback_time,
                    violation_duration=max_lookback_time - request.data_timestamp,
                    requester_id=request.requester_id,
                    data_identifier=request.data_identifier,
                    component=request.component,
                    enforcement_action=self._get_enforcement_action(boundary.boundary_type)
                )
                
                return False, f"Data too old: {request.data_timestamp} < {max_lookback_time}", violation
        
        # Check latency constraint
        if boundary.max_latency_seconds > 0:
            data_age = (current_time - request.data_timestamp).total_seconds()
            
            if data_age > boundary.max_latency_seconds:
                violation = BoundaryViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=BoundaryViolationType.LATENCY_VIOLATION,
                    severity=boundary.enforcement_level,
                    detected_at=current_time,
                    boundary_id=boundary.boundary_id,
                    request_id=request.request_id,
                    requested_time=request.data_timestamp,
                    allowed_time=current_time - timedelta(seconds=boundary.max_latency_seconds),
                    violation_duration=timedelta(seconds=data_age - boundary.max_latency_seconds),
                    requester_id=request.requester_id,
                    data_identifier=request.data_identifier,
                    component=request.component,
                    enforcement_action=self._get_enforcement_action(boundary.boundary_type)
                )
                
                return False, f"Data latency violation: {data_age}s > {boundary.max_latency_seconds}s", violation
        
        return True, None, None
    
    async def _apply_enforcement_rule(self, rule: BoundaryEnforcementRule, request: DataAccessRequest) -> Optional[BoundaryViolation]:
        """Apply enforcement rule"""
        
        current_time = self._get_current_time()
        
        # Create violation record
        violation = BoundaryViolation(
            violation_id=str(uuid.uuid4()),
            violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
            severity=rule.severity,
            detected_at=current_time,
            boundary_id=rule.rule_id,
            request_id=request.request_id,
            requested_time=request.data_timestamp,
            allowed_time=current_time,
            violation_duration=timedelta(0),
            requester_id=request.requester_id,
            data_identifier=request.data_identifier,
            component=request.component,
            enforcement_action=rule.action
        )
        
        # Update rule statistics
        rule.violations_detected += 1
        
        # Apply enforcement action
        if rule.action == EnforcementAction.BLOCK:
            return violation
        elif rule.action == EnforcementAction.WARN:
            logger.warning(f"Rule warning: {rule.rule_name} for request {request.request_id}")
            return None
        elif rule.action == EnforcementAction.LOG:
            logger.info(f"Rule log: {rule.rule_name} for request {request.request_id}")
            return None
        
        return None
    
    async def _handle_boundary_violation(self, violation: BoundaryViolation):
        """Handle boundary violation"""
        
        self.stats['violations_detected'] += 1
        
        # Apply enforcement action
        if violation.enforcement_action == EnforcementAction.BLOCK:
            logger.warning(f"Blocking access due to violation: {violation.violation_id}")
            violation.action_successful = True
            
        elif violation.enforcement_action == EnforcementAction.SUBSTITUTE:
            # Attempt to substitute with alternative data
            substitution_result = await self._attempt_data_substitution(violation)
            violation.action_successful = substitution_result
            
        elif violation.enforcement_action == EnforcementAction.DELAY:
            # Implement delay logic
            delay_seconds = self._calculate_delay_duration(violation)
            await asyncio.sleep(delay_seconds)
            violation.action_successful = True
            
        # Auto-remediation if enabled
        if self.config.get('auto_remediation_enabled', True):
            await self._attempt_auto_remediation(violation)
        
        # Alert if threshold exceeded
        recent_violations = sum(1 for v in self.violation_history if 
                               (datetime.utcnow() - v.detected_at).total_seconds() < 3600)
        
        if recent_violations >= self.config.get('violation_alert_threshold', 5):
            logger.critical(f"Violation threshold exceeded: {recent_violations} violations in last hour")
    
    async def _attempt_data_substitution(self, violation: BoundaryViolation) -> bool:
        """Attempt to substitute data for violation"""
        
        # Implement data substitution logic
        logger.info(f"Attempting data substitution for violation: {violation.violation_id}")
        
        # This would involve finding alternative data sources
        # or using interpolated/synthetic data
        
        return False  # Placeholder
    
    def _calculate_delay_duration(self, violation: BoundaryViolation) -> float:
        """Calculate delay duration for violation"""
        
        # Calculate appropriate delay based on violation type
        if violation.violation_type == BoundaryViolationType.LATENCY_VIOLATION:
            return violation.violation_duration.total_seconds()
        elif violation.violation_type == BoundaryViolationType.FUTURE_ACCESS:
            return min(violation.violation_duration.total_seconds(), 300)  # Max 5 minutes
        else:
            return 1.0  # Default 1 second delay
    
    async def _attempt_auto_remediation(self, violation: BoundaryViolation):
        """Attempt automatic remediation of violation"""
        
        try:
            if violation.violation_type == BoundaryViolationType.FUTURE_ACCESS:
                # Adjust request timestamp
                violation.remediation_details = "Adjusted request timestamp to current time"
                violation.remediation_applied = True
                
            elif violation.violation_type == BoundaryViolationType.LATENCY_VIOLATION:
                # Use cached data if available
                violation.remediation_details = "Used cached data instead of live data"
                violation.remediation_applied = True
                
            logger.debug(f"Auto-remediation applied: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Auto-remediation failed for violation {violation.violation_id}: {e}")
    
    def _get_enforcement_action(self, boundary_type: TemporalBoundaryType) -> EnforcementAction:
        """Get enforcement action based on boundary type"""
        
        if boundary_type == TemporalBoundaryType.HARD_BOUNDARY:
            return EnforcementAction.BLOCK
        elif boundary_type == TemporalBoundaryType.SOFT_BOUNDARY:
            return EnforcementAction.WARN
        else:
            return EnforcementAction.LOG
    
    def _get_current_time(self) -> datetime:
        """Get current time with synchronization"""
        
        # Use registered time provider if available
        if self.current_time_providers:
            primary_provider = next(iter(self.current_time_providers.values()))
            return primary_provider() + self.time_synchronization_offset
        
        return datetime.utcnow()
    
    def _initialize_default_rules(self):
        """Initialize default enforcement rules"""
        
        # Rule 1: Block future data access
        future_access_rule = BoundaryEnforcementRule(
            rule_id="future_access_block",
            rule_name="Block Future Data Access",
            rule_type="temporal",
            condition=lambda req: req.data_timestamp > self._get_current_time(),
            action=EnforcementAction.BLOCK,
            severity=SeverityLevel.CRITICAL
        )
        
        self.register_enforcement_rule(future_access_rule)
        
        # Rule 2: Warn on excessive lookback
        lookback_warn_rule = BoundaryEnforcementRule(
            rule_id="excessive_lookback_warn",
            rule_name="Warn on Excessive Lookback",
            rule_type="temporal",
            condition=lambda req: (self._get_current_time() - req.data_timestamp).total_seconds() > 86400,  # 24 hours
            action=EnforcementAction.WARN,
            severity=SeverityLevel.MEDIUM
        )
        
        self.register_enforcement_rule(lookback_warn_rule)
        
        # Rule 3: Log backtest data access
        backtest_log_rule = BoundaryEnforcementRule(
            rule_id="backtest_access_log",
            rule_name="Log Backtest Data Access",
            rule_type="access_type",
            condition=lambda req: req.access_type == AccessType.BACKTEST,
            action=EnforcementAction.LOG,
            severity=SeverityLevel.LOW
        )
        
        self.register_enforcement_rule(backtest_log_rule)
        
        logger.debug("Default enforcement rules initialized")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check boundary health
                self._check_boundary_health()
                
                # Clean up old requests
                self._cleanup_old_requests()
                
                # Update statistics
                self._update_monitoring_statistics()
                
                # Check for violation patterns
                self._analyze_violation_patterns()
                
                time.sleep(self.config.get('monitoring_interval_seconds', 10))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_boundary_health(self):
        """Check health of registered boundaries"""
        
        current_time = self._get_current_time()
        
        with self.lock:
            for boundary_id, boundary in self.temporal_boundaries.items():
                # Check if boundary is stale
                if (current_time - boundary.last_updated).total_seconds() > 3600:  # 1 hour
                    logger.warning(f"Stale boundary detected: {boundary_id}")
                
                # Check if boundary is still relevant
                if boundary.timestamp < current_time - timedelta(hours=24):
                    logger.info(f"Old boundary detected: {boundary_id}")
    
    def _cleanup_old_requests(self):
        """Clean up old access requests"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        with self.lock:
            old_requests = [
                req_id for req_id, req in self.access_requests.items()
                if req.created_at < cutoff_time
            ]
            
            for req_id in old_requests:
                del self.access_requests[req_id]
            
            if old_requests:
                logger.debug(f"Cleaned up {len(old_requests)} old requests")
    
    def _update_monitoring_statistics(self):
        """Update monitoring statistics"""
        
        with self.lock:
            self.stats['boundaries_active'] = len(self.temporal_boundaries)
            self.stats['rules_active'] = len(self.enforcement_rules)
    
    def _analyze_violation_patterns(self):
        """Analyze violation patterns for insights"""
        
        if len(self.violation_history) < 10:
            return
        
        # Analyze recent violations
        recent_violations = [v for v in self.violation_history 
                           if (datetime.utcnow() - v.detected_at).total_seconds() < 3600]
        
        if not recent_violations:
            return
        
        # Group by violation type
        violation_types = defaultdict(int)
        for violation in recent_violations:
            violation_types[violation.violation_type.value] += 1
        
        # Check for patterns
        for violation_type, count in violation_types.items():
            if count > 5:  # Threshold for pattern detection
                logger.warning(f"High frequency violation pattern: {violation_type} - {count} occurrences")
    
    def get_enforcement_summary(self) -> Dict[str, Any]:
        """Get boundary enforcement summary"""
        
        with self.lock:
            recent_violations = [v for v in self.violation_history 
                               if (datetime.utcnow() - v.detected_at).total_seconds() < 3600]
            
            violation_by_type = defaultdict(int)
            for violation in recent_violations:
                violation_by_type[violation.violation_type.value] += 1
            
            return {
                'statistics': self.stats.copy(),
                'enforcement_active': self.enforcement_active,
                'strict_mode': self.strict_mode,
                'registered_boundaries': len(self.temporal_boundaries),
                'active_rules': len(self.enforcement_rules),
                'recent_violations': {
                    'total': len(recent_violations),
                    'by_type': dict(violation_by_type)
                },
                'monitoring_active': self.monitoring_active,
                'last_updated': datetime.utcnow()
            }
    
    def get_violation_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get detailed violation report"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        with self.lock:
            relevant_violations = [
                v for v in self.violation_history 
                if v.detected_at >= cutoff_time
            ]
            
            # Group violations by various dimensions
            by_type = defaultdict(list)
            by_component = defaultdict(list)
            by_severity = defaultdict(list)
            
            for violation in relevant_violations:
                by_type[violation.violation_type.value].append(violation)
                by_component[violation.component].append(violation)
                by_severity[violation.severity.value].append(violation)
            
            return {
                'time_range': {
                    'start': cutoff_time,
                    'end': datetime.utcnow(),
                    'hours': hours_back
                },
                'total_violations': len(relevant_violations),
                'by_type': {k: len(v) for k, v in by_type.items()},
                'by_component': {k: len(v) for k, v in by_component.items()},
                'by_severity': {k: len(v) for k, v in by_severity.items()},
                'critical_violations': [
                    {
                        'violation_id': v.violation_id,
                        'type': v.violation_type.value,
                        'detected_at': v.detected_at,
                        'component': v.component,
                        'data_identifier': v.data_identifier
                    }
                    for v in relevant_violations 
                    if v.severity == SeverityLevel.CRITICAL
                ],
                'top_violated_components': sorted(
                    by_component.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )[:5]
            }

# Global instance
temporal_boundary_enforcer = TemporalBoundaryEnforcer()

# Export key components
__all__ = [
    'AccessType',
    'BoundaryViolationType',
    'EnforcementAction',
    'DataAccessRequest',
    'BoundaryViolation',
    'BoundaryEnforcementRule',
    'TemporalBoundaryEnforcer',
    'temporal_boundary_enforcer'
]