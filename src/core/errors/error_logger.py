"""
Structured Error Logging and Reporting System

Provides comprehensive error logging with structured data, metrics collection,
and reporting capabilities.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import threading
from pathlib import Path

from .base_exceptions import BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext

logger = logging.getLogger(__name__)


@dataclass
class ErrorReport:
    """Structured error report."""
    timestamp: float
    error_type: str
    error_message: str
    error_code: str
    severity: str
    category: str
    correlation_id: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    resolution_status: str = "unresolved"
    resolution_time: Optional[float] = None
    resolution_notes: Optional[str] = None


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring and analysis."""
    total_errors: int = 0
    errors_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0
    avg_resolution_time: float = 0.0
    uptime: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def update_error_rate(self, window_size: int = 3600):
        """Update error rate per hour."""
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        if elapsed_hours > 0:
            self.error_rate = self.total_errors / elapsed_hours
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return {
            "total_errors": self.total_errors,
            "errors_by_severity": dict(self.errors_by_severity),
            "errors_by_category": dict(self.errors_by_category),
            "errors_by_type": dict(self.errors_by_type),
            "error_rate": self.error_rate,
            "avg_resolution_time": self.avg_resolution_time,
            "uptime": time.time() - self.start_time
        }


class ErrorLoggerConfig:
    """Configuration for error logger."""
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_file: Optional[str] = None,
        max_log_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_structured_logging: bool = True,
        enable_metrics: bool = True,
        metrics_window_size: int = 1000,
        enable_error_aggregation: bool = True,
        aggregation_window: int = 300  # 5 minutes
    ):
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_structured_logging = enable_structured_logging
        self.enable_metrics = enable_metrics
        self.metrics_window_size = metrics_window_size
        self.enable_error_aggregation = enable_error_aggregation
        self.aggregation_window = aggregation_window


class StructuredErrorLogger:
    """Structured error logger with metrics and reporting."""
    
    def __init__(self, config: ErrorLoggerConfig):
        self.config = config
        self.metrics = ErrorMetrics()
        self.reports: List[ErrorReport] = []
        self.recent_errors: deque = deque(maxlen=config.metrics_window_size)
        self.error_aggregator = ErrorAggregator() if config.enable_error_aggregation else None
        self._lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger("grandmodel.errors")
        self.logger.setLevel(self.config.log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.log_level)
            formatter = logging.Formatter(self.config.log_format)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(self.config.log_level)
            formatter = logging.Formatter(self.config.log_format)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_error(
        self,
        error: BaseGrandModelError,
        additional_context: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = True
    ) -> str:
        """Log structured error and return report ID."""
        with self._lock:
            # Create error report
            report = ErrorReport(
                timestamp=time.time(),
                error_type=type(error).__name__,
                error_message=error.message,
                error_code=error.error_code,
                severity=error.severity.value,
                category=error.category.value,
                correlation_id=error.context.correlation_id,
                context=asdict(error.context),
                error_details=error.error_details
            )
            
            # Add stack trace if requested
            if include_stack_trace:
                import traceback
                report.stack_trace = traceback.format_exc()
            
            # Add additional context
            if additional_context:
                report.context.update(additional_context)
            
            # Store report
            self.reports.append(report)
            self.recent_errors.append(report)
            
            # Update metrics
            self._update_metrics(report)
            
            # Log to aggregator
            if self.error_aggregator:
                self.error_aggregator.add_error(report)
            
            # Log structured message
            if self.config.enable_structured_logging:
                self._log_structured_message(report)
            else:
                self._log_simple_message(report)
            
            return report.correlation_id
    
    def _update_metrics(self, report: ErrorReport):
        """Update error metrics."""
        if not self.config.enable_metrics:
            return
        
        self.metrics.total_errors += 1
        self.metrics.errors_by_severity[report.severity] += 1
        self.metrics.errors_by_category[report.category] += 1
        self.metrics.errors_by_type[report.error_type] += 1
        self.metrics.update_error_rate()
    
    def _log_structured_message(self, report: ErrorReport):
        """Log structured error message."""
        log_data = {
            "timestamp": report.timestamp,
            "error_type": report.error_type,
            "error_code": report.error_code,
            "severity": report.severity,
            "category": report.category,
            "correlation_id": report.correlation_id,
            "message": report.error_message,
            "context": report.context,
            "error_details": report.error_details
        }
        
        # Log based on severity
        if report.severity == "critical":
            self.logger.critical(json.dumps(log_data, indent=2))
        elif report.severity == "high":
            self.logger.error(json.dumps(log_data, indent=2))
        elif report.severity == "medium":
            self.logger.warning(json.dumps(log_data, indent=2))
        else:
            self.logger.info(json.dumps(log_data, indent=2))
    
    def _log_simple_message(self, report: ErrorReport):
        """Log simple error message."""
        message = f"[{report.error_code}] {report.error_message} (correlation_id: {report.correlation_id})"
        
        if report.severity == "critical":
            self.logger.critical(message)
        elif report.severity == "high":
            self.logger.error(message)
        elif report.severity == "medium":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        with self._lock:
            return self.metrics.get_metrics_dict()
    
    def get_recent_errors(self, limit: int = 100) -> List[ErrorReport]:
        """Get recent error reports."""
        with self._lock:
            return list(self.recent_errors)[-limit:]
    
    def get_errors_by_severity(self, severity: str) -> List[ErrorReport]:
        """Get errors by severity level."""
        with self._lock:
            return [r for r in self.reports if r.severity == severity]
    
    def get_errors_by_category(self, category: str) -> List[ErrorReport]:
        """Get errors by category."""
        with self._lock:
            return [r for r in self.reports if r.category == category]
    
    def get_errors_by_timerange(self, start_time: float, end_time: float) -> List[ErrorReport]:
        """Get errors within time range."""
        with self._lock:
            return [r for r in self.reports if start_time <= r.timestamp <= end_time]
    
    def mark_error_resolved(self, correlation_id: str, resolution_notes: str = ""):
        """Mark error as resolved."""
        with self._lock:
            for report in self.reports:
                if report.correlation_id == correlation_id:
                    report.resolution_status = "resolved"
                    report.resolution_time = time.time()
                    report.resolution_notes = resolution_notes
                    break
    
    def export_reports(self, file_path: str, format: str = "json"):
        """Export error reports to file."""
        with self._lock:
            reports_data = [asdict(report) for report in self.reports]
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(reports_data, f, indent=2)
            elif format == "csv":
                import csv
                if reports_data:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=reports_data[0].keys())
                        writer.writeheader()
                        writer.writerows(reports_data)
    
    def clear_old_reports(self, days: int = 30):
        """Clear old error reports."""
        cutoff_time = time.time() - (days * 24 * 3600)
        with self._lock:
            self.reports = [r for r in self.reports if r.timestamp > cutoff_time]


class ErrorAggregator:
    """Aggregates similar errors to reduce noise."""
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_windows: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = threading.Lock()
    
    def add_error(self, report: ErrorReport):
        """Add error to aggregation."""
        with self._lock:
            key = f"{report.error_type}:{report.error_code}"
            current_time = time.time()
            
            # Clean old entries
            self._clean_old_entries(key, current_time)
            
            # Add new entry
            self.error_windows[key].append(current_time)
            self.error_counts[key] += 1
    
    def _clean_old_entries(self, key: str, current_time: float):
        """Clean old entries from window."""
        window = self.error_windows[key]
        while window and window[0] < current_time - self.window_size:
            window.popleft()
            self.error_counts[key] -= 1
    
    def get_error_frequency(self, error_type: str, error_code: str) -> int:
        """Get error frequency in current window."""
        with self._lock:
            key = f"{error_type}:{error_code}"
            self._clean_old_entries(key, time.time())
            return self.error_counts[key]
    
    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top errors by frequency."""
        with self._lock:
            current_time = time.time()
            
            # Clean all entries
            for key in list(self.error_counts.keys()):
                self._clean_old_entries(key, current_time)
            
            # Sort by count
            sorted_errors = sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            return [
                {
                    "error_key": key,
                    "count": count,
                    "frequency": count / (self.window_size / 60)  # per minute
                }
                for key, count in sorted_errors
                if count > 0
            ]


class ErrorLogger:
    """Main error logger interface."""
    
    def __init__(self, config: Optional[ErrorLoggerConfig] = None):
        self.config = config or ErrorLoggerConfig()
        self.structured_logger = StructuredErrorLogger(self.config)
    
    def log_error(
        self,
        error: Union[BaseGrandModelError, Exception],
        context: Optional[ErrorContext] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = True
    ) -> str:
        """Log error with full context."""
        
        # Convert non-GrandModel exceptions
        if not isinstance(error, BaseGrandModelError):
            error = BaseGrandModelError(
                message=str(error),
                context=context or ErrorContext(),
                cause=error
            )
        
        return self.structured_logger.log_error(
            error=error,
            additional_context=additional_context,
            include_stack_trace=include_stack_trace
        )
    
    def log_exception(
        self,
        message: str,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM
    ) -> str:
        """Log exception with custom message."""
        error = BaseGrandModelError(
            message=message,
            severity=severity,
            category=category,
            context=context or ErrorContext(),
            cause=exception
        )
        
        return self.structured_logger.log_error(error)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error metrics."""
        return self.structured_logger.get_metrics()
    
    def get_recent_errors(self, limit: int = 100) -> List[ErrorReport]:
        """Get recent errors."""
        return self.structured_logger.get_recent_errors(limit)
    
    def export_reports(self, file_path: str, format: str = "json"):
        """Export error reports."""
        self.structured_logger.export_reports(file_path, format)


# Global error logger instance
_global_error_logger = None
_logger_lock = threading.Lock()


def get_global_error_logger() -> ErrorLogger:
    """Get global error logger instance."""
    global _global_error_logger
    
    if _global_error_logger is None:
        with _logger_lock:
            if _global_error_logger is None:
                _global_error_logger = ErrorLogger()
    
    return _global_error_logger


def log_error(
    error: Union[BaseGrandModelError, Exception],
    context: Optional[ErrorContext] = None,
    **kwargs
) -> str:
    """Log error using global logger."""
    return get_global_error_logger().log_error(error, context, **kwargs)


def log_exception(
    message: str,
    exception: Exception,
    context: Optional[ErrorContext] = None,
    **kwargs
) -> str:
    """Log exception using global logger."""
    return get_global_error_logger().log_exception(message, exception, context, **kwargs)