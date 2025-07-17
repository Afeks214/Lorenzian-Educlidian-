"""
Audit System for Comprehensive Activity Tracking

This module provides comprehensive audit logging, trail management, and
accountability tracking for all trading system activities.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
from pathlib import Path
import sqlite3
import uuid
import threading
from contextlib import contextmanager

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATED = "position_updated"
    RISK_BREACH = "risk_breach"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_CONFIGURATION = "system_configuration"
    DATA_ACCESS = "data_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SECURITY_EVENT = "security_event"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    ALERT_TRIGGERED = "alert_triggered"
    MANUAL_INTERVENTION = "manual_intervention"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(Enum):
    """Outcomes of audited actions"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class AuditContext:
    """Context information for audit events"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    system_component: str
    action: str
    resource: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Represents a single audit event"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    outcome: AuditOutcome
    timestamp: datetime
    context: AuditContext
    description: str
    details: Dict[str, Any]
    checksum: str = field(default="")
    previous_event_id: Optional[str] = None
    chain_sequence: int = 0
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "timestamp": self.timestamp.isoformat(),
            "context": asdict(self.context),
            "description": self.description,
            "details": self.details,
            "previous_event_id": self.previous_event_id,
            "chain_sequence": self.chain_sequence
        }
        
        serialized = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        calculated_checksum = self._calculate_checksum()
        return calculated_checksum == self.checksum


@dataclass
class AuditTrail:
    """Represents a chain of related audit events"""
    trail_id: str
    trail_name: str
    start_time: datetime
    end_time: Optional[datetime]
    events: List[AuditEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: AuditEvent):
        """Add event to trail"""
        if self.events:
            event.previous_event_id = self.events[-1].event_id
            event.chain_sequence = len(self.events)
        
        self.events.append(event)
    
    def verify_chain_integrity(self) -> bool:
        """Verify integrity of entire event chain"""
        if not self.events:
            return True
        
        # Check first event
        first_event = self.events[0]
        if first_event.previous_event_id is not None or first_event.chain_sequence != 0:
            return False
        
        # Check chain consistency
        for i, event in enumerate(self.events):
            if not event.verify_integrity():
                return False
            
            if i > 0:
                if event.previous_event_id != self.events[i-1].event_id:
                    return False
                if event.chain_sequence != i:
                    return False
        
        return True


class AuditStorage:
    """Storage backend for audit events"""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize audit database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Audit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    system_component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    previous_event_id TEXT,
                    chain_sequence INTEGER NOT NULL,
                    additional_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit trails table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trails (
                    trail_id TEXT PRIMARY KEY,
                    trail_name TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit trail events junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail_events (
                    trail_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    PRIMARY KEY (trail_id, event_id),
                    FOREIGN KEY (trail_id) REFERENCES audit_trails (trail_id),
                    FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                )
            """)
            
            # Audit summaries table for performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_summaries (
                    summary_id TEXT PRIMARY KEY,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    total_events INTEGER NOT NULL,
                    events_by_type TEXT NOT NULL,
                    events_by_severity TEXT NOT NULL,
                    events_by_outcome TEXT NOT NULL,
                    top_users TEXT NOT NULL,
                    top_components TEXT NOT NULL,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_system_component ON audit_events(system_component)")
            
            conn.commit()
    
    def store_event(self, event: AuditEvent) -> bool:
        """Store audit event in database"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO audit_events (
                            event_id, event_type, severity, outcome, timestamp,
                            user_id, session_id, ip_address, user_agent,
                            system_component, action, resource, description,
                            details, checksum, previous_event_id, chain_sequence,
                            additional_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.outcome.value,
                        event.timestamp,
                        event.context.user_id,
                        event.context.session_id,
                        event.context.ip_address,
                        event.context.user_agent,
                        event.context.system_component,
                        event.context.action,
                        event.context.resource,
                        event.description,
                        json.dumps(event.details),
                        event.checksum,
                        event.previous_event_id,
                        event.chain_sequence,
                        json.dumps(event.context.additional_data)
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error("Failed to store audit event", event_id=event.event_id, error=str(e))
            return False
    
    def store_trail(self, trail: AuditTrail) -> bool:
        """Store audit trail in database"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Store trail metadata
                    cursor.execute("""
                        INSERT OR REPLACE INTO audit_trails (
                            trail_id, trail_name, start_time, end_time, metadata
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        trail.trail_id,
                        trail.trail_name,
                        trail.start_time,
                        trail.end_time,
                        json.dumps(trail.metadata)
                    ))
                    
                    # Store trail-event relationships
                    cursor.execute("DELETE FROM audit_trail_events WHERE trail_id = ?", (trail.trail_id,))
                    
                    for i, event in enumerate(trail.events):
                        cursor.execute("""
                            INSERT INTO audit_trail_events (trail_id, event_id, sequence_number)
                            VALUES (?, ?, ?)
                        """, (trail.trail_id, event.event_id, i))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error("Failed to store audit trail", trail_id=trail.trail_id, error=str(e))
            return False
    
    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        system_component: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity.value)
                
                if system_component:
                    query += " AND system_component = ?"
                    params.append(system_component)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    context = AuditContext(
                        user_id=row[5],
                        session_id=row[6],
                        ip_address=row[7],
                        user_agent=row[8] or "",
                        system_component=row[9],
                        action=row[10],
                        resource=row[11],
                        additional_data=json.loads(row[17]) if row[17] else {}
                    )
                    
                    event = AuditEvent(
                        event_id=row[0],
                        event_type=AuditEventType(row[1]),
                        severity=AuditSeverity(row[2]),
                        outcome=AuditOutcome(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                        context=context,
                        description=row[12],
                        details=json.loads(row[13]),
                        checksum=row[14],
                        previous_event_id=row[15],
                        chain_sequence=row[16]
                    )
                    
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error("Failed to retrieve audit events", error=str(e))
            return []
    
    def get_trail(self, trail_id: str) -> Optional[AuditTrail]:
        """Retrieve audit trail by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trail metadata
                cursor.execute("SELECT * FROM audit_trails WHERE trail_id = ?", (trail_id,))
                trail_row = cursor.fetchone()
                
                if not trail_row:
                    return None
                
                # Get trail events
                cursor.execute("""
                    SELECT ae.* FROM audit_events ae
                    JOIN audit_trail_events ate ON ae.event_id = ate.event_id
                    WHERE ate.trail_id = ?
                    ORDER BY ate.sequence_number
                """, (trail_id,))
                
                event_rows = cursor.fetchall()
                
                events = []
                for row in event_rows:
                    context = AuditContext(
                        user_id=row[5],
                        session_id=row[6],
                        ip_address=row[7],
                        user_agent=row[8] or "",
                        system_component=row[9],
                        action=row[10],
                        resource=row[11],
                        additional_data=json.loads(row[17]) if row[17] else {}
                    )
                    
                    event = AuditEvent(
                        event_id=row[0],
                        event_type=AuditEventType(row[1]),
                        severity=AuditSeverity(row[2]),
                        outcome=AuditOutcome(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                        context=context,
                        description=row[12],
                        details=json.loads(row[13]),
                        checksum=row[14],
                        previous_event_id=row[15],
                        chain_sequence=row[16]
                    )
                    
                    events.append(event)
                
                trail = AuditTrail(
                    trail_id=trail_row[0],
                    trail_name=trail_row[1],
                    start_time=datetime.fromisoformat(trail_row[2]),
                    end_time=datetime.fromisoformat(trail_row[3]) if trail_row[3] else None,
                    events=events,
                    metadata=json.loads(trail_row[4]) if trail_row[4] else {}
                )
                
                return trail
                
        except Exception as e:
            logger.error("Failed to retrieve audit trail", trail_id=trail_id, error=str(e))
            return None


class AuditSystem:
    """Main audit system for comprehensive activity tracking"""
    
    def __init__(self, event_bus: EventBus, storage: Optional[AuditStorage] = None):
        self.event_bus = event_bus
        self.storage = storage or AuditStorage()
        
        # Current audit context
        self.current_context: Optional[AuditContext] = None
        
        # Active audit trails
        self.active_trails: Dict[str, AuditTrail] = {}
        
        # Event counters
        self.event_count = 0
        self.trail_count = 0
        
        # Performance metrics
        self.storage_errors = 0
        self.integrity_failures = 0
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize system
        self._initialize_system()
        
        logger.info("Audit System initialized")
    
    def _setup_event_handlers(self):
        """Setup event handlers for automatic audit logging"""
        self.event_bus.subscribe(EventType.EXECUTE_TRADE, self._handle_trade_event)
        self.event_bus.subscribe(EventType.ORDER_PLACED, self._handle_order_event)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_event)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_event)
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_strategic_event)
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._handle_error_event)
    
    def _initialize_system(self):
        """Initialize audit system"""
        # Log system start
        system_context = AuditContext(
            user_id="system",
            session_id="system_session",
            ip_address="127.0.0.1",
            user_agent="audit_system",
            system_component="audit_system",
            action="system_start",
            resource="audit_system"
        )
        
        self.log_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            context=system_context,
            description="Audit system started",
            details={"version": "1.0.0", "timestamp": datetime.now().isoformat()}
        )
    
    @contextmanager
    def audit_context(self, context: AuditContext):
        """Context manager for setting audit context"""
        old_context = self.current_context
        self.current_context = context
        try:
            yield
        finally:
            self.current_context = old_context
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        outcome: AuditOutcome,
        context: Optional[AuditContext] = None,
        description: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit event"""
        try:
            # Use provided context or current context
            if context is None:
                context = self.current_context
            
            if context is None:
                # Create default context
                context = AuditContext(
                    user_id="unknown",
                    session_id="unknown",
                    ip_address="unknown",
                    user_agent="unknown",
                    system_component="unknown",
                    action="unknown",
                    resource="unknown"
                )
            
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                outcome=outcome,
                timestamp=datetime.now(),
                context=context,
                description=description,
                details=details or {}
            )
            
            # Store event
            success = self.storage.store_event(event)
            if not success:
                self.storage_errors += 1
                logger.error("Failed to store audit event", event_id=event_id)
            
            # Verify integrity
            if not event.verify_integrity():
                self.integrity_failures += 1
                logger.error("Audit event integrity check failed", event_id=event_id)
            
            self.event_count += 1
            
            # Publish audit event
            self._publish_audit_event(event)
            
            logger.debug("Audit event logged", event_id=event_id, event_type=event_type.value)
            
            return event_id
            
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
            return ""
    
    def _publish_audit_event(self, event: AuditEvent):
        """Publish audit event to event bus"""
        try:
            event_payload = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "outcome": event.outcome.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.context.user_id,
                "system_component": event.context.system_component,
                "action": event.context.action,
                "description": event.description,
                "details": event.details
            }
            
            audit_event = self.event_bus.create_event(
                event_type=EventType.AUDIT_EVENT,  # Assuming this exists
                payload=event_payload,
                source="audit_system"
            )
            
            self.event_bus.publish(audit_event)
            
        except Exception as e:
            logger.error("Failed to publish audit event", error=str(e))
    
    def start_trail(self, trail_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new audit trail"""
        try:
            trail_id = str(uuid.uuid4())
            
            trail = AuditTrail(
                trail_id=trail_id,
                trail_name=trail_name,
                start_time=datetime.now(),
                end_time=None,
                events=[],
                metadata=metadata or {}
            )
            
            self.active_trails[trail_id] = trail
            self.trail_count += 1
            
            # Log trail start
            if self.current_context:
                self.log_event(
                    event_type=AuditEventType.MANUAL_INTERVENTION,
                    severity=AuditSeverity.INFO,
                    outcome=AuditOutcome.SUCCESS,
                    description=f"Audit trail started: {trail_name}",
                    details={"trail_id": trail_id, "trail_name": trail_name}
                )
            
            logger.info("Audit trail started", trail_id=trail_id, trail_name=trail_name)
            
            return trail_id
            
        except Exception as e:
            logger.error("Failed to start audit trail", trail_name=trail_name, error=str(e))
            return ""
    
    def end_trail(self, trail_id: str) -> bool:
        """End an audit trail"""
        try:
            if trail_id not in self.active_trails:
                logger.warning("Attempted to end non-existent trail", trail_id=trail_id)
                return False
            
            trail = self.active_trails[trail_id]
            trail.end_time = datetime.now()
            
            # Verify trail integrity
            if not trail.verify_chain_integrity():
                self.integrity_failures += 1
                logger.error("Audit trail integrity check failed", trail_id=trail_id)
            
            # Store trail
            success = self.storage.store_trail(trail)
            if not success:
                self.storage_errors += 1
                logger.error("Failed to store audit trail", trail_id=trail_id)
            
            # Remove from active trails
            del self.active_trails[trail_id]
            
            # Log trail end
            if self.current_context:
                self.log_event(
                    event_type=AuditEventType.MANUAL_INTERVENTION,
                    severity=AuditSeverity.INFO,
                    outcome=AuditOutcome.SUCCESS,
                    description=f"Audit trail ended: {trail.trail_name}",
                    details={"trail_id": trail_id, "event_count": len(trail.events)}
                )
            
            logger.info("Audit trail ended", trail_id=trail_id, event_count=len(trail.events))
            
            return True
            
        except Exception as e:
            logger.error("Failed to end audit trail", trail_id=trail_id, error=str(e))
            return False
    
    def log_to_trail(self, trail_id: str, event_id: str) -> bool:
        """Add an event to an audit trail"""
        try:
            if trail_id not in self.active_trails:
                logger.warning("Attempted to log to non-existent trail", trail_id=trail_id)
                return False
            
            # Get the event
            events = self.storage.get_events(limit=1)  # This is inefficient, but for demo
            event = next((e for e in events if e.event_id == event_id), None)
            
            if not event:
                logger.warning("Event not found for trail", event_id=event_id, trail_id=trail_id)
                return False
            
            # Add to trail
            trail = self.active_trails[trail_id]
            trail.add_event(event)
            
            logger.debug("Event added to audit trail", event_id=event_id, trail_id=trail_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to log event to trail", event_id=event_id, trail_id=trail_id, error=str(e))
            return False
    
    def _handle_trade_event(self, event: Event):
        """Handle trade execution events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "trading_system"),
                system_component="trading_engine",
                action="execute_trade",
                resource=payload.get("symbol", "unknown")
            )
            
            self.log_event(
                event_type=AuditEventType.TRADE_EXECUTED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=context,
                description=f"Trade executed: {payload.get('symbol', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling trade event for audit", error=str(e))
    
    def _handle_order_event(self, event: Event):
        """Handle order placement events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "trading_system"),
                system_component="order_management",
                action="place_order",
                resource=payload.get("symbol", "unknown")
            )
            
            self.log_event(
                event_type=AuditEventType.ORDER_PLACED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=context,
                description=f"Order placed: {payload.get('symbol', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling order event for audit", error=str(e))
    
    def _handle_position_event(self, event: Event):
        """Handle position update events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "trading_system"),
                system_component="position_manager",
                action="update_position",
                resource=payload.get("symbol", "unknown")
            )
            
            self.log_event(
                event_type=AuditEventType.POSITION_UPDATED,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=context,
                description=f"Position updated: {payload.get('symbol', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling position event for audit", error=str(e))
    
    def _handle_risk_event(self, event: Event):
        """Handle risk breach events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "risk_system"),
                system_component="risk_engine",
                action="risk_breach",
                resource=payload.get("resource", "portfolio")
            )
            
            # Determine event type based on payload
            if payload.get("type") == "policy_violation":
                event_type = AuditEventType.POLICY_VIOLATION
            elif payload.get("type") == "compliance_violation":
                event_type = AuditEventType.COMPLIANCE_VIOLATION
            else:
                event_type = AuditEventType.RISK_BREACH
            
            self.log_event(
                event_type=event_type,
                severity=AuditSeverity.WARNING,
                outcome=AuditOutcome.FAILURE,
                context=context,
                description=f"Risk breach detected: {payload.get('message', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling risk event for audit", error=str(e))
    
    def _handle_strategic_event(self, event: Event):
        """Handle strategic decision events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "strategic_system"),
                system_component="strategic_engine",
                action="strategic_decision",
                resource=payload.get("resource", "portfolio")
            )
            
            self.log_event(
                event_type=AuditEventType.MANUAL_INTERVENTION,
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
                context=context,
                description=f"Strategic decision made: {payload.get('description', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling strategic event for audit", error=str(e))
    
    def _handle_error_event(self, event: Event):
        """Handle system error events"""
        try:
            payload = event.payload
            
            context = AuditContext(
                user_id=payload.get("user_id", "system"),
                session_id=payload.get("session_id", "system_session"),
                ip_address=payload.get("ip_address", "unknown"),
                user_agent=payload.get("user_agent", "system"),
                system_component=payload.get("component", "unknown"),
                action="error_occurred",
                resource=payload.get("resource", "system")
            )
            
            self.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                severity=AuditSeverity.ERROR,
                outcome=AuditOutcome.FAILURE,
                context=context,
                description=f"System error: {payload.get('message', 'unknown')}",
                details=payload
            )
            
        except Exception as e:
            logger.error("Error handling error event for audit", error=str(e))
    
    def get_audit_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit summary statistics"""
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=7)
            
            # Get events for period
            events = self.storage.get_events(start_time=start_time, end_time=end_time)
            
            # Calculate statistics
            total_events = len(events)
            
            events_by_type = {}
            events_by_severity = {}
            events_by_outcome = {}
            users = set()
            components = set()
            
            for event in events:
                # Count by type
                events_by_type[event.event_type.value] = events_by_type.get(event.event_type.value, 0) + 1
                
                # Count by severity
                events_by_severity[event.severity.value] = events_by_severity.get(event.severity.value, 0) + 1
                
                # Count by outcome
                events_by_outcome[event.outcome.value] = events_by_outcome.get(event.outcome.value, 0) + 1
                
                # Track users and components
                users.add(event.context.user_id)
                components.add(event.context.system_component)
            
            return {
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "total_events": total_events,
                "events_by_type": events_by_type,
                "events_by_severity": events_by_severity,
                "events_by_outcome": events_by_outcome,
                "unique_users": len(users),
                "unique_components": len(components),
                "storage_errors": self.storage_errors,
                "integrity_failures": self.integrity_failures,
                "active_trails": len(self.active_trails)
            }
            
        except Exception as e:
            logger.error("Failed to generate audit summary", error=str(e))
            return {}
    
    def verify_audit_integrity(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify audit log integrity"""
        try:
            events = self.storage.get_events(start_time=start_time, end_time=end_time)
            
            total_events = len(events)
            integrity_failures = 0
            corrupted_events = []
            
            for event in events:
                if not event.verify_integrity():
                    integrity_failures += 1
                    corrupted_events.append(event.event_id)
            
            integrity_score = (total_events - integrity_failures) / total_events if total_events > 0 else 1.0
            
            return {
                "total_events": total_events,
                "integrity_failures": integrity_failures,
                "corrupted_events": corrupted_events,
                "integrity_score": integrity_score,
                "is_tampered": integrity_failures > 0
            }
            
        except Exception as e:
            logger.error("Failed to verify audit integrity", error=str(e))
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get audit system status"""
        return {
            "total_events": self.event_count,
            "total_trails": self.trail_count,
            "active_trails": len(self.active_trails),
            "storage_errors": self.storage_errors,
            "integrity_failures": self.integrity_failures,
            "storage_path": str(self.storage.db_path)
        }
    
    def shutdown(self):
        """Shutdown audit system"""
        try:
            # End all active trails
            for trail_id in list(self.active_trails.keys()):
                self.end_trail(trail_id)
            
            # Log system shutdown
            if self.current_context:
                self.log_event(
                    event_type=AuditEventType.SYSTEM_STOP,
                    severity=AuditSeverity.INFO,
                    outcome=AuditOutcome.SUCCESS,
                    description="Audit system shutdown",
                    details={"total_events": self.event_count}
                )
            
            logger.info("Audit system shutdown complete")
            
        except Exception as e:
            logger.error("Error during audit system shutdown", error=str(e))