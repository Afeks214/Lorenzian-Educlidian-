"""
Comprehensive error monitoring, logging, and pattern detection system.

Provides advanced error tracking, pattern detection, alerting, and metrics
collection for the trading system error handling infrastructure.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
import re
from pathlib import Path

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext
)
from .agent_error_decorators import AgentType

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorPattern(Enum):
    """Types of error patterns"""
    FREQUENCY_SPIKE = "frequency_spike"
    CASCADING_FAILURE = "cascading_failure"
    RECURRING_ERROR = "recurring_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class ErrorEvent:
    """Represents a single error event"""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    agent_type: Optional[AgentType] = None
    agent_id: Optional[str] = None
    context: Optional[ErrorContext] = None
    stack_trace: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'agent_type': self.agent_type.value if self.agent_type else None,
            'agent_id': self.agent_id,
            'context': self.context.to_dict() if self.context else None,
            'stack_trace': self.stack_trace,
            'additional_data': self.additional_data
        }


@dataclass
class ErrorPatternAlert:
    """Alert for detected error patterns"""
    pattern_type: ErrorPattern
    severity: AlertLevel
    description: str
    timestamp: datetime
    affected_agents: List[str]
    error_count: int
    pattern_details: Dict[str, Any]
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'affected_agents': self.affected_agents,
            'error_count': self.error_count,
            'pattern_details': self.pattern_details,
            'recommended_actions': self.recommended_actions
        }


@dataclass
class ErrorMetrics:
    """Aggregated error metrics"""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_agent: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    average_resolution_time: float = 0.0
    pattern_alerts: List[ErrorPatternAlert] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_errors': self.total_errors,
            'errors_by_type': self.errors_by_type,
            'errors_by_severity': self.errors_by_severity,
            'errors_by_category': self.errors_by_category,
            'errors_by_agent': self.errors_by_agent,
            'error_rate': self.error_rate,
            'average_resolution_time': self.average_resolution_time,
            'pattern_alerts': [alert.to_dict() for alert in self.pattern_alerts]
        }


class ErrorPatternDetector:
    """Detects patterns in error events"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.frequency_threshold = self.config.get('frequency_threshold', 10)
        self.time_window = self.config.get('time_window', 300)  # 5 minutes
        self.cascading_threshold = self.config.get('cascading_threshold', 5)
        self.performance_threshold = self.config.get('performance_threshold', 2.0)
        
    def detect_patterns(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect error patterns from event list"""
        alerts = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Detect frequency spikes
        alerts.extend(self._detect_frequency_spikes(sorted_events))
        
        # Detect cascading failures
        alerts.extend(self._detect_cascading_failures(sorted_events))
        
        # Detect recurring errors
        alerts.extend(self._detect_recurring_errors(sorted_events))
        
        # Detect performance degradation
        alerts.extend(self._detect_performance_degradation(sorted_events))
        
        # Detect dependency failures
        alerts.extend(self._detect_dependency_failures(sorted_events))
        
        return alerts
    
    def _detect_frequency_spikes(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect frequency spikes in errors"""
        alerts = []
        
        if len(events) < self.frequency_threshold:
            return alerts
        
        # Group events by time windows
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.time_window)
        recent_events = [e for e in events if e.timestamp >= window_start]
        
        if len(recent_events) >= self.frequency_threshold:
            # Group by error type
            error_type_counts = defaultdict(int)
            for event in recent_events:
                error_type_counts[event.error_type] += 1
            
            for error_type, count in error_type_counts.items():
                if count >= self.frequency_threshold:
                    alerts.append(ErrorPatternAlert(
                        pattern_type=ErrorPattern.FREQUENCY_SPIKE,
                        severity=AlertLevel.WARNING,
                        description=f"Frequency spike detected: {count} {error_type} errors in {self.time_window} seconds",
                        timestamp=current_time,
                        affected_agents=list(set(e.agent_id for e in recent_events if e.agent_id and e.error_type == error_type)),
                        error_count=count,
                        pattern_details={
                            'error_type': error_type,
                            'time_window': self.time_window,
                            'threshold': self.frequency_threshold
                        },
                        recommended_actions=[
                            f"Investigate root cause of {error_type} errors",
                            "Check system resources and dependencies",
                            "Consider implementing circuit breakers"
                        ]
                    ))
        
        return alerts
    
    def _detect_cascading_failures(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect cascading failures across agents"""
        alerts = []
        
        # Group events by agent
        agent_events = defaultdict(list)
        for event in events:
            if event.agent_id:
                agent_events[event.agent_id].append(event)
        
        # Look for sequential failures across agents
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.time_window)
        
        # Check if multiple agents failed in sequence
        failed_agents = []
        for agent_id, agent_events_list in agent_events.items():
            recent_failures = [e for e in agent_events_list if e.timestamp >= window_start and e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]]
            if len(recent_failures) >= self.cascading_threshold:
                failed_agents.append(agent_id)
        
        if len(failed_agents) >= 3:  # Multiple agents failing
            alerts.append(ErrorPatternAlert(
                pattern_type=ErrorPattern.CASCADING_FAILURE,
                severity=AlertLevel.CRITICAL,
                description=f"Cascading failure detected across {len(failed_agents)} agents",
                timestamp=current_time,
                affected_agents=failed_agents,
                error_count=sum(len(agent_events[agent]) for agent in failed_agents),
                pattern_details={
                    'failed_agents': failed_agents,
                    'time_window': self.time_window
                },
                recommended_actions=[
                    "Implement emergency circuit breakers",
                    "Check system-wide dependencies",
                    "Consider system shutdown if necessary",
                    "Investigate shared resource issues"
                ]
            ))
        
        return alerts
    
    def _detect_recurring_errors(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect recurring error patterns"""
        alerts = []
        
        # Group by error message patterns
        error_patterns = defaultdict(list)
        for event in events:
            # Simple pattern matching - can be enhanced with regex
            pattern_key = self._extract_error_pattern(event.error_message)
            error_patterns[pattern_key].append(event)
        
        current_time = datetime.utcnow()
        for pattern, pattern_events in error_patterns.items():
            if len(pattern_events) >= 5:  # Recurring threshold
                # Check if they're spread over time (not just a burst)
                timestamps = [e.timestamp for e in pattern_events]
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                
                if time_span > 3600:  # Spread over more than 1 hour
                    alerts.append(ErrorPatternAlert(
                        pattern_type=ErrorPattern.RECURRING_ERROR,
                        severity=AlertLevel.WARNING,
                        description=f"Recurring error pattern detected: {pattern}",
                        timestamp=current_time,
                        affected_agents=list(set(e.agent_id for e in pattern_events if e.agent_id)),
                        error_count=len(pattern_events),
                        pattern_details={
                            'pattern': pattern,
                            'time_span_hours': time_span / 3600,
                            'first_occurrence': min(timestamps).isoformat(),
                            'last_occurrence': max(timestamps).isoformat()
                        },
                        recommended_actions=[
                            "Investigate root cause of recurring pattern",
                            "Check for memory leaks or resource issues",
                            "Review error handling logic",
                            "Consider implementing preventive measures"
                        ]
                    ))
        
        return alerts
    
    def _detect_performance_degradation(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect performance degradation patterns"""
        alerts = []
        
        # Look for timeout and performance-related errors
        performance_events = [e for e in events if 'timeout' in e.error_message.lower() or 
                             e.category == ErrorCategory.PERFORMANCE]
        
        if len(performance_events) >= 5:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=self.time_window)
            recent_performance_events = [e for e in performance_events if e.timestamp >= window_start]
            
            if len(recent_performance_events) >= 3:
                alerts.append(ErrorPatternAlert(
                    pattern_type=ErrorPattern.PERFORMANCE_DEGRADATION,
                    severity=AlertLevel.ERROR,
                    description=f"Performance degradation detected: {len(recent_performance_events)} performance-related errors",
                    timestamp=current_time,
                    affected_agents=list(set(e.agent_id for e in recent_performance_events if e.agent_id)),
                    error_count=len(recent_performance_events),
                    pattern_details={
                        'performance_errors': len(recent_performance_events),
                        'time_window': self.time_window
                    },
                    recommended_actions=[
                        "Check system resources (CPU, memory, disk)",
                        "Analyze network latency",
                        "Review database performance",
                        "Consider scaling system resources"
                    ]
                ))
        
        return alerts
    
    def _detect_dependency_failures(self, events: List[ErrorEvent]) -> List[ErrorPatternAlert]:
        """Detect dependency failure patterns"""
        alerts = []
        
        # Look for dependency-related errors
        dependency_events = [e for e in events if e.category == ErrorCategory.DEPENDENCY or
                           e.category == ErrorCategory.NETWORK or
                           e.category == ErrorCategory.EXTERNAL_SERVICE]
        
        if len(dependency_events) >= 3:
            current_time = datetime.utcnow()
            
            # Group by dependency type
            dependency_groups = defaultdict(list)
            for event in dependency_events:
                # Extract dependency name from error message or context
                dependency_name = self._extract_dependency_name(event)
                dependency_groups[dependency_name].append(event)
            
            for dependency, dep_events in dependency_groups.items():
                if len(dep_events) >= 3:
                    alerts.append(ErrorPatternAlert(
                        pattern_type=ErrorPattern.DEPENDENCY_FAILURE,
                        severity=AlertLevel.ERROR,
                        description=f"Dependency failure detected: {dependency}",
                        timestamp=current_time,
                        affected_agents=list(set(e.agent_id for e in dep_events if e.agent_id)),
                        error_count=len(dep_events),
                        pattern_details={
                            'dependency': dependency,
                            'failure_count': len(dep_events)
                        },
                        recommended_actions=[
                            f"Check {dependency} service health",
                            "Implement circuit breaker for dependency",
                            "Enable fallback mechanisms",
                            "Contact dependency provider if external"
                        ]
                    ))
        
        return alerts
    
    def _extract_error_pattern(self, error_message: str) -> str:
        """Extract error pattern from error message"""
        # Simple pattern extraction - replace specific values with placeholders
        pattern = re.sub(r'\d+', 'NUM', error_message)
        pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', pattern)
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 'TIMESTAMP', pattern)
        return pattern
    
    def _extract_dependency_name(self, event: ErrorEvent) -> str:
        """Extract dependency name from error event"""
        # Simple extraction - can be enhanced
        if event.additional_data and 'dependency' in event.additional_data:
            return event.additional_data['dependency']
        
        # Try to extract from error message
        message = event.error_message.lower()
        if 'market data' in message:
            return 'market_data'
        elif 'execution' in message:
            return 'execution_venue'
        elif 'risk' in message:
            return 'risk_system'
        elif 'database' in message:
            return 'database'
        else:
            return 'unknown'


class ErrorMonitoringSystem:
    """Comprehensive error monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_events: deque = deque(maxlen=self.config.get('max_events', 10000))
        self.pattern_detector = ErrorPatternDetector(self.config.get('pattern_detection', {}))
        self.alert_handlers: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_running = False
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # 1 minute
        
        # File logging
        self.log_file_path = self.config.get('log_file_path')
        if self.log_file_path:
            self.log_file_path = Path(self.log_file_path)
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Error monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Error monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_running:
            try:
                time.sleep(self.monitoring_interval)
                self._analyze_patterns()
                self._generate_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def record_error(self, error: Exception, context: Optional[ErrorContext] = None,
                    agent_type: Optional[AgentType] = None, agent_id: Optional[str] = None,
                    additional_data: Optional[Dict[str, Any]] = None):
        """Record an error event"""
        
        # Create error event
        if isinstance(error, BaseGrandModelError):
            severity = error.severity
            category = error.category
            error_context = error.context or context
        else:
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.SYSTEM
            error_context = context
        
        event = ErrorEvent(
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            agent_type=agent_type,
            agent_id=agent_id,
            context=error_context,
            stack_trace=None,  # Can be added if needed
            additional_data=additional_data or {}
        )
        
        with self.lock:
            self.error_events.append(event)
        
        # Log to file if configured
        if self.log_file_path:
            self._log_to_file(event)
        
        # Trigger immediate analysis for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._analyze_patterns()
        
        logger.debug(f"Error recorded: {event.error_type} - {event.error_message}")
    
    def _log_to_file(self, event: ErrorEvent):
        """Log error event to file"""
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")
    
    def _analyze_patterns(self):
        """Analyze error patterns and generate alerts"""
        with self.lock:
            events = list(self.error_events)
        
        if not events:
            return
        
        # Detect patterns
        alerts = self.pattern_detector.detect_patterns(events)
        
        # Handle alerts
        for alert in alerts:
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: ErrorPatternAlert):
        """Handle a pattern alert"""
        logger.warning(f"Error pattern alert: {alert.description}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _generate_metrics(self):
        """Generate and publish metrics"""
        with self.lock:
            events = list(self.error_events)
        
        if not events:
            return
        
        # Calculate metrics
        metrics = self._calculate_metrics(events)
        
        # Notify metrics callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def _calculate_metrics(self, events: List[ErrorEvent]) -> ErrorMetrics:
        """Calculate error metrics"""
        metrics = ErrorMetrics()
        
        metrics.total_errors = len(events)
        
        # Group by various dimensions
        for event in events:
            metrics.errors_by_type[event.error_type] = metrics.errors_by_type.get(event.error_type, 0) + 1
            metrics.errors_by_severity[event.severity.value] = metrics.errors_by_severity.get(event.severity.value, 0) + 1
            metrics.errors_by_category[event.category.value] = metrics.errors_by_category.get(event.category.value, 0) + 1
            
            if event.agent_id:
                metrics.errors_by_agent[event.agent_id] = metrics.errors_by_agent.get(event.agent_id, 0) + 1
        
        # Calculate error rate (errors per minute)
        if events:
            time_span = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds()
            if time_span > 0:
                metrics.error_rate = len(events) / (time_span / 60)  # errors per minute
        
        # Get recent pattern alerts
        recent_alerts = self.pattern_detector.detect_patterns(events)
        metrics.pattern_alerts = recent_alerts
        
        return metrics
    
    def register_alert_handler(self, handler: Callable[[ErrorPatternAlert], None]):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def register_metrics_callback(self, callback: Callable[[ErrorMetrics], None]):
        """Register a metrics callback"""
        self.metrics_callbacks.append(callback)
    
    def get_current_metrics(self) -> ErrorMetrics:
        """Get current error metrics"""
        with self.lock:
            events = list(self.error_events)
        
        return self._calculate_metrics(events)
    
    def get_error_history(self, hours: int = 24) -> List[ErrorEvent]:
        """Get error history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.lock:
            return [event for event in self.error_events if event.timestamp >= cutoff_time]
    
    def get_agent_error_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get error statistics for a specific agent"""
        with self.lock:
            agent_events = [event for event in self.error_events if event.agent_id == agent_id]
        
        if not agent_events:
            return {'total_errors': 0}
        
        stats = {
            'total_errors': len(agent_events),
            'errors_by_type': {},
            'errors_by_severity': {},
            'first_error': min(e.timestamp for e in agent_events).isoformat(),
            'last_error': max(e.timestamp for e in agent_events).isoformat()
        }
        
        for event in agent_events:
            stats['errors_by_type'][event.error_type] = stats['errors_by_type'].get(event.error_type, 0) + 1
            stats['errors_by_severity'][event.severity.value] = stats['errors_by_severity'].get(event.severity.value, 0) + 1
        
        return stats
    
    def clear_history(self):
        """Clear error history"""
        with self.lock:
            self.error_events.clear()
        logger.info("Error history cleared")
    
    def export_error_data(self, output_file: str, hours: int = 24):
        """Export error data to file"""
        error_history = self.get_error_history(hours)
        
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'hours_covered': hours,
            'total_errors': len(error_history),
            'errors': [event.to_dict() for event in error_history]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Error data exported to {output_file}")
    
    def __del__(self):
        """Destructor"""
        self.stop_monitoring()


# Global error monitoring system
_global_error_monitor = ErrorMonitoringSystem()


def get_error_monitor() -> ErrorMonitoringSystem:
    """Get global error monitoring system"""
    return _global_error_monitor


def record_error(error: Exception, context: Optional[ErrorContext] = None,
                agent_type: Optional[AgentType] = None, agent_id: Optional[str] = None,
                additional_data: Optional[Dict[str, Any]] = None):
    """Record an error in the global monitoring system"""
    _global_error_monitor.record_error(error, context, agent_type, agent_id, additional_data)


def get_error_metrics() -> ErrorMetrics:
    """Get current error metrics"""
    return _global_error_monitor.get_current_metrics()


def export_error_report(output_file: str, hours: int = 24):
    """Export error report to file"""
    _global_error_monitor.export_error_data(output_file, hours)