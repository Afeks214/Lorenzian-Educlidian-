#!/usr/bin/env python3
"""
AGENT 5 MISSION: Switch Event Logger
Comprehensive audit trail for system state changes

This module provides specialized audit logging for system switch events with:
- State change tracking
- Command audit trail
- Performance logging
- Compliance reporting
- Security monitoring
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import sqlite3
from pathlib import Path
import logging

# Import existing audit components
try:
    from .audit_logger import (
        EnhancedAuditLogger, 
        AuditMetadata, 
        AuditCategory, 
        AuditLevel,
        ComplianceContext,
        ComplianceFramework,
        create_trading_metadata
    )
    from ..core.event_bus import EventBus
    from ..utils.logger import get_logger
    AUDIT_AVAILABLE = True
except ImportError:
    # Fallback for basic logging
    AUDIT_AVAILABLE = False
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class SystemState(Enum):
    """System state enumeration"""
    OFF = "OFF"
    ON = "ON"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

class CommandType(Enum):
    """Command type enumeration"""
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    STATUS = "status"
    EMERGENCY_STOP = "emergency_stop"
    HEALTH_CHECK = "health_check"

class EventType(Enum):
    """Event type enumeration"""
    STATE_CHANGE = "state_change"
    COMMAND_EXECUTION = "command_execution"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"

@dataclass
class StateChangeEvent:
    """State change event details"""
    event_id: str
    timestamp: datetime
    previous_state: SystemState
    new_state: SystemState
    trigger: str
    user_id: str
    session_id: str
    command_type: Optional[CommandType] = None
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommandExecutionEvent:
    """Command execution event details"""
    event_id: str
    timestamp: datetime
    command_type: CommandType
    user_id: str
    session_id: str
    ip_address: str
    arguments: Dict[str, Any]
    execution_time: float
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    system_state_before: Optional[SystemState] = None
    system_state_after: Optional[SystemState] = None

@dataclass
class PerformanceEvent:
    """Performance monitoring event"""
    event_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    component: str
    threshold_breached: bool = False
    severity: str = "info"
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

class SwitchEventLogger:
    """Specialized event logger for system switch operations"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path("data/switch_events.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize audit logger if available
        if AUDIT_AVAILABLE:
            self.audit_logger = EnhancedAuditLogger()
        else:
            self.audit_logger = None
        
        # Event storage
        self.event_history = []
        self.state_changes = []
        self.command_history = []
        self.performance_events = []
        
        # Event counters
        self.event_counters = {
            'state_changes': 0,
            'commands_executed': 0,
            'errors_encountered': 0,
            'performance_events': 0
        }
        
        # Initialize database
        self._initialize_database()
        
        logger.info("Switch event logger initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for event storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # State change events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS state_change_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        previous_state TEXT NOT NULL,
                        new_state TEXT NOT NULL,
                        trigger TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        command_type TEXT,
                        duration REAL DEFAULT 0.0,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT,
                        additional_context TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Command execution events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS command_execution_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        command_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        ip_address TEXT NOT NULL,
                        arguments TEXT,
                        execution_time REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        result TEXT,
                        error_message TEXT,
                        system_state_before TEXT,
                        system_state_after TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        component TEXT NOT NULL,
                        threshold_breached BOOLEAN DEFAULT FALSE,
                        severity TEXT DEFAULT 'info',
                        additional_metrics TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_state_change_timestamp ON state_change_events(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_state_change_user ON state_change_events(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_state_change_states ON state_change_events(previous_state, new_state)",
                    "CREATE INDEX IF NOT EXISTS idx_command_execution_timestamp ON command_execution_events(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_command_execution_type ON command_execution_events(command_type)",
                    "CREATE INDEX IF NOT EXISTS idx_command_execution_user ON command_execution_events(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_events(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_component ON performance_events(component)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_metric ON performance_events(metric_name)"
                ]
                
                for index in indexes:
                    cursor.execute(index)
                
                conn.commit()
                logger.info("Event logger database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def log_state_change(
        self,
        previous_state: SystemState,
        new_state: SystemState,
        trigger: str,
        user_id: str = "system",
        session_id: str = "default",
        command_type: Optional[CommandType] = None,
        duration: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log system state change event"""
        try:
            event_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create state change event
            event = StateChangeEvent(
                event_id=event_id,
                timestamp=timestamp,
                previous_state=previous_state,
                new_state=new_state,
                trigger=trigger,
                user_id=user_id,
                session_id=session_id,
                command_type=command_type,
                duration=duration,
                success=success,
                error_message=error_message,
                additional_context=additional_context or {}
            )
            
            # Store in database
            self._store_state_change_event(event)
            
            # Store in memory
            self.state_changes.append(event)
            self.state_changes = self.state_changes[-100:]  # Keep last 100
            
            # Update counters
            self.event_counters['state_changes'] += 1
            if not success:
                self.event_counters['errors_encountered'] += 1
            
            # Log to audit system if available
            if self.audit_logger:
                metadata = create_trading_metadata(user_id, session_id, "127.0.0.1")
                
                # Determine audit level
                audit_level = AuditLevel.ERROR if not success else AuditLevel.INFO
                if new_state == SystemState.EMERGENCY_STOP:
                    audit_level = AuditLevel.CRITICAL
                
                self.audit_logger.log_event(
                    event_type="system_state_change",
                    category=AuditCategory.OPERATIONAL,
                    level=audit_level,
                    metadata=metadata,
                    message=f"System state changed from {previous_state.value} to {new_state.value}",
                    details={
                        'previous_state': previous_state.value,
                        'new_state': new_state.value,
                        'trigger': trigger,
                        'duration': duration,
                        'success': success,
                        'error_message': error_message,
                        'additional_context': additional_context
                    },
                    regulatory_impact=new_state in [SystemState.ON, SystemState.EMERGENCY_STOP],
                    financial_impact=new_state in [SystemState.ON, SystemState.OFF, SystemState.EMERGENCY_STOP]
                )
            
            logger.info(f"State change logged: {previous_state.value} -> {new_state.value}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log state change: {e}")
            return ""
    
    def log_command_execution(
        self,
        command_type: CommandType,
        user_id: str,
        session_id: str,
        ip_address: str,
        arguments: Dict[str, Any],
        execution_time: float,
        success: bool,
        result: Optional[Any] = None,
        error_message: Optional[str] = None,
        system_state_before: Optional[SystemState] = None,
        system_state_after: Optional[SystemState] = None
    ) -> str:
        """Log command execution event"""
        try:
            event_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create command execution event
            event = CommandExecutionEvent(
                event_id=event_id,
                timestamp=timestamp,
                command_type=command_type,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                arguments=arguments,
                execution_time=execution_time,
                success=success,
                result=result,
                error_message=error_message,
                system_state_before=system_state_before,
                system_state_after=system_state_after
            )
            
            # Store in database
            self._store_command_execution_event(event)
            
            # Store in memory
            self.command_history.append(event)
            self.command_history = self.command_history[-100:]  # Keep last 100
            
            # Update counters
            self.event_counters['commands_executed'] += 1
            if not success:
                self.event_counters['errors_encountered'] += 1
            
            # Log to audit system if available
            if self.audit_logger:
                metadata = create_trading_metadata(user_id, session_id, ip_address)
                
                # Determine audit level
                audit_level = AuditLevel.ERROR if not success else AuditLevel.INFO
                if command_type == CommandType.EMERGENCY_STOP:
                    audit_level = AuditLevel.CRITICAL
                
                self.audit_logger.log_event(
                    event_type="command_execution",
                    category=AuditCategory.OPERATIONAL,
                    level=audit_level,
                    metadata=metadata,
                    message=f"Command executed: {command_type.value}",
                    details={
                        'command_type': command_type.value,
                        'arguments': arguments,
                        'execution_time': execution_time,
                        'success': success,
                        'result': str(result) if result else None,
                        'error_message': error_message,
                        'system_state_before': system_state_before.value if system_state_before else None,
                        'system_state_after': system_state_after.value if system_state_after else None
                    },
                    regulatory_impact=command_type in [CommandType.TURN_ON, CommandType.EMERGENCY_STOP],
                    financial_impact=command_type in [CommandType.TURN_ON, CommandType.TURN_OFF, CommandType.EMERGENCY_STOP]
                )
            
            logger.info(f"Command execution logged: {command_type.value} ({execution_time:.2f}s)")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log command execution: {e}")
            return ""
    
    def log_performance_event(
        self,
        metric_name: str,
        metric_value: float,
        component: str,
        threshold_breached: bool = False,
        severity: str = "info",
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log performance monitoring event"""
        try:
            event_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create performance event
            event = PerformanceEvent(
                event_id=event_id,
                timestamp=timestamp,
                metric_name=metric_name,
                metric_value=metric_value,
                component=component,
                threshold_breached=threshold_breached,
                severity=severity,
                additional_metrics=additional_metrics or {}
            )
            
            # Store in database
            self._store_performance_event(event)
            
            # Store in memory
            self.performance_events.append(event)
            self.performance_events = self.performance_events[-1000:]  # Keep last 1000
            
            # Update counters
            self.event_counters['performance_events'] += 1
            
            # Log to audit system if available (only for threshold breaches)
            if self.audit_logger and threshold_breached:
                metadata = create_trading_metadata("system", "performance_monitor", "127.0.0.1")
                
                # Determine audit level based on severity
                audit_level_map = {
                    "info": AuditLevel.INFO,
                    "warning": AuditLevel.WARNING,
                    "error": AuditLevel.ERROR,
                    "critical": AuditLevel.CRITICAL
                }
                audit_level = audit_level_map.get(severity, AuditLevel.INFO)
                
                self.audit_logger.log_event(
                    event_type="performance_threshold_breach",
                    category=AuditCategory.OPERATIONAL,
                    level=audit_level,
                    metadata=metadata,
                    message=f"Performance threshold breached: {metric_name} = {metric_value}",
                    details={
                        'metric_name': metric_name,
                        'metric_value': metric_value,
                        'component': component,
                        'severity': severity,
                        'additional_metrics': additional_metrics
                    }
                )
            
            if threshold_breached:
                logger.warning(f"Performance threshold breached: {component}.{metric_name} = {metric_value}")
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log performance event: {e}")
            return ""
    
    def _store_state_change_event(self, event: StateChangeEvent):
        """Store state change event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO state_change_events (
                        event_id, timestamp, previous_state, new_state, trigger,
                        user_id, session_id, command_type, duration, success,
                        error_message, additional_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp,
                    event.previous_state.value,
                    event.new_state.value,
                    event.trigger,
                    event.user_id,
                    event.session_id,
                    event.command_type.value if event.command_type else None,
                    event.duration,
                    event.success,
                    event.error_message,
                    json.dumps(event.additional_context)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store state change event: {e}")
    
    def _store_command_execution_event(self, event: CommandExecutionEvent):
        """Store command execution event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO command_execution_events (
                        event_id, timestamp, command_type, user_id, session_id,
                        ip_address, arguments, execution_time, success, result,
                        error_message, system_state_before, system_state_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp,
                    event.command_type.value,
                    event.user_id,
                    event.session_id,
                    event.ip_address,
                    json.dumps(event.arguments),
                    event.execution_time,
                    event.success,
                    json.dumps(event.result) if event.result else None,
                    event.error_message,
                    event.system_state_before.value if event.system_state_before else None,
                    event.system_state_after.value if event.system_state_after else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store command execution event: {e}")
    
    def _store_performance_event(self, event: PerformanceEvent):
        """Store performance event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_events (
                        event_id, timestamp, metric_name, metric_value, component,
                        threshold_breached, severity, additional_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp,
                    event.metric_name,
                    event.metric_value,
                    event.component,
                    event.threshold_breached,
                    event.severity,
                    json.dumps(event.additional_metrics)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store performance event: {e}")
    
    def get_state_change_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get state change history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM state_change_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get state change history: {e}")
            return []
    
    def get_command_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get command execution history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM command_execution_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get command history: {e}")
            return []
    
    def get_performance_history(self, component: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance event history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if component:
                    cursor.execute("""
                        SELECT * FROM performance_events 
                        WHERE component = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (component, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM performance_events 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get counts
                cursor.execute("SELECT COUNT(*) FROM state_change_events")
                state_changes = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM command_execution_events")
                commands = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM performance_events")
                performance = cursor.fetchone()[0]
                
                # Get error counts
                cursor.execute("SELECT COUNT(*) FROM state_change_events WHERE success = FALSE")
                state_errors = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM command_execution_events WHERE success = FALSE")
                command_errors = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM performance_events WHERE threshold_breached = TRUE")
                performance_breaches = cursor.fetchone()[0]
                
                return {
                    'total_events': state_changes + commands + performance,
                    'state_changes': state_changes,
                    'commands_executed': commands,
                    'performance_events': performance,
                    'state_change_errors': state_errors,
                    'command_errors': command_errors,
                    'performance_breaches': performance_breaches,
                    'counters': self.event_counters
                }
                
        except Exception as e:
            logger.error(f"Failed to get event statistics: {e}")
            return {}
    
    def generate_activity_report(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate activity report for specified time period"""
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # State changes in period
                cursor.execute("""
                    SELECT COUNT(*) FROM state_change_events 
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_time, end_time))
                state_changes = cursor.fetchone()[0]
                
                # Commands in period
                cursor.execute("""
                    SELECT COUNT(*) FROM command_execution_events 
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_time, end_time))
                commands = cursor.fetchone()[0]
                
                # Performance events in period
                cursor.execute("""
                    SELECT COUNT(*) FROM performance_events 
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_time, end_time))
                performance = cursor.fetchone()[0]
                
                # Error counts
                cursor.execute("""
                    SELECT COUNT(*) FROM state_change_events 
                    WHERE timestamp BETWEEN ? AND ? AND success = FALSE
                """, (start_time, end_time))
                state_errors = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM command_execution_events 
                    WHERE timestamp BETWEEN ? AND ? AND success = FALSE
                """, (start_time, end_time))
                command_errors = cursor.fetchone()[0]
                
                # Most common commands
                cursor.execute("""
                    SELECT command_type, COUNT(*) as count 
                    FROM command_execution_events 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY command_type 
                    ORDER BY count DESC
                """, (start_time, end_time))
                command_breakdown = dict(cursor.fetchall())
                
                # Most common state changes
                cursor.execute("""
                    SELECT previous_state, new_state, COUNT(*) as count 
                    FROM state_change_events 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY previous_state, new_state 
                    ORDER BY count DESC
                """, (start_time, end_time))
                state_transitions = [
                    {'from': row[0], 'to': row[1], 'count': row[2]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    'report_period': {
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'duration_hours': (end_time - start_time).total_seconds() / 3600
                    },
                    'summary': {
                        'total_events': state_changes + commands + performance,
                        'state_changes': state_changes,
                        'commands_executed': commands,
                        'performance_events': performance,
                        'total_errors': state_errors + command_errors,
                        'error_rate': (state_errors + command_errors) / max(1, state_changes + commands)
                    },
                    'breakdowns': {
                        'commands': command_breakdown,
                        'state_transitions': state_transitions
                    },
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to generate activity report: {e}")
            return {}
    
    def cleanup_old_events(self, retention_days: int = 30):
        """Clean up old events beyond retention period"""
        try:
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old events
                cursor.execute("DELETE FROM state_change_events WHERE timestamp < ?", (cutoff_time,))
                state_deleted = cursor.rowcount
                
                cursor.execute("DELETE FROM command_execution_events WHERE timestamp < ?", (cutoff_time,))
                command_deleted = cursor.rowcount
                
                cursor.execute("DELETE FROM performance_events WHERE timestamp < ?", (cutoff_time,))
                performance_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old events: {state_deleted} state changes, {command_deleted} commands, {performance_deleted} performance events")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")

# Factory function
def create_switch_event_logger(db_path: Optional[str] = None) -> SwitchEventLogger:
    """Create switch event logger instance"""
    return SwitchEventLogger(db_path)

# Example usage
if __name__ == "__main__":
    logger = create_switch_event_logger()
    
    # Test state change logging
    logger.log_state_change(
        previous_state=SystemState.OFF,
        new_state=SystemState.ON,
        trigger="Manual start",
        user_id="admin",
        session_id="test_session"
    )
    
    # Test command logging
    logger.log_command_execution(
        command_type=CommandType.TURN_ON,
        user_id="admin",
        session_id="test_session",
        ip_address="127.0.0.1",
        arguments={"verbose": True},
        execution_time=2.5,
        success=True
    )
    
    # Get statistics
    stats = logger.get_event_statistics()
    print(json.dumps(stats, indent=2))