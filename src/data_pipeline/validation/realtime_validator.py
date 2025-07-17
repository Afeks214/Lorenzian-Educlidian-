"""
Real-time data validation and quality checks with sub-millisecond performance

This module implements comprehensive data validation and quality monitoring
for high-frequency market data processing.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import statistics
import math
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult(Enum):
    """Validation result types"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

@dataclass
class ValidationIssue:
    """Data validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any
    timestamp: float = field(default_factory=time.time)
    rule_name: str = ""
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ValidationMetrics:
    """Validation performance metrics"""
    total_validations: int = 0
    total_failures: int = 0
    total_warnings: int = 0
    avg_validation_time_us: float = 0.0
    max_validation_time_us: float = 0.0
    throughput_validations_per_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    timestamp: float
    total_records: int
    valid_records: int
    invalid_records: int
    quality_score: float  # 0-100
    issues: List[ValidationIssue]
    field_statistics: Dict[str, Dict[str, Any]]
    validation_metrics: ValidationMetrics

class ValidationRule:
    """Base class for validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
        self.enabled = True
        self.call_count = 0
        self.failure_count = 0
        self.total_time_us = 0.0
    
    def validate(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        """Validate data and return result with optional error message"""
        if not self.enabled:
            return ValidationResult.SKIP, None
        
        start_time = time.time_ns()
        
        try:
            self.call_count += 1
            result, message = self._validate_impl(data, field, metadata)
            
            if result == ValidationResult.FAIL:
                self.failure_count += 1
            
            return result, message
        
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Error in validation rule {self.name}: {str(e)}")
            return ValidationResult.FAIL, f"Validation error: {str(e)}"
        
        finally:
            end_time = time.time_ns()
            self.total_time_us += (end_time - start_time) / 1000
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        """Implementation of validation logic - override in subclasses"""
        raise NotImplementedError
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this rule"""
        avg_time_us = self.total_time_us / self.call_count if self.call_count > 0 else 0
        failure_rate = self.failure_count / self.call_count if self.call_count > 0 else 0
        
        return {
            'name': self.name,
            'call_count': self.call_count,
            'failure_count': self.failure_count,
            'failure_rate': failure_rate,
            'avg_time_us': avg_time_us,
            'total_time_us': self.total_time_us
        }

class RangeValidationRule(ValidationRule):
    """Validate numeric values within a range"""
    
    def __init__(self, name: str, min_value: Optional[float] = None, max_value: Optional[float] = None, 
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.min_value = min_value
        self.max_value = max_value
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, (int, float, np.number)):
            return ValidationResult.FAIL, f"Expected numeric value, got {type(data)}"
        
        value = float(data)
        
        if math.isnan(value) or math.isinf(value):
            return ValidationResult.FAIL, f"Invalid numeric value: {value}"
        
        if self.min_value is not None and value < self.min_value:
            return ValidationResult.FAIL, f"Value {value} below minimum {self.min_value}"
        
        if self.max_value is not None and value > self.max_value:
            return ValidationResult.FAIL, f"Value {value} above maximum {self.max_value}"
        
        return ValidationResult.PASS, None

class NotNullValidationRule(ValidationRule):
    """Validate that values are not null/None/NaN"""
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if data is None:
            return ValidationResult.FAIL, "Value is None"
        
        if isinstance(data, float) and math.isnan(data):
            return ValidationResult.FAIL, "Value is NaN"
        
        if isinstance(data, str) and data.strip() == "":
            return ValidationResult.FAIL, "Value is empty string"
        
        return ValidationResult.PASS, None

class TypeValidationRule(ValidationRule):
    """Validate data type"""
    
    def __init__(self, name: str, expected_type: type, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.expected_type = expected_type
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, self.expected_type):
            return ValidationResult.FAIL, f"Expected {self.expected_type.__name__}, got {type(data).__name__}"
        
        return ValidationResult.PASS, None

class PriceValidationRule(ValidationRule):
    """Specialized validation for price data"""
    
    def __init__(self, name: str, min_price: float = 0.0, max_price: float = 1000000.0,
                 max_change_percent: float = 20.0, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.min_price = min_price
        self.max_price = max_price
        self.max_change_percent = max_change_percent
        self.price_history = deque(maxlen=100)
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, (int, float, np.number)):
            return ValidationResult.FAIL, f"Expected numeric price, got {type(data)}"
        
        price = float(data)
        
        # Check for invalid numbers
        if math.isnan(price) or math.isinf(price):
            return ValidationResult.FAIL, f"Invalid price value: {price}"
        
        # Check range
        if price < self.min_price or price > self.max_price:
            return ValidationResult.FAIL, f"Price {price} outside valid range [{self.min_price}, {self.max_price}]"
        
        # Check for extreme price changes
        if len(self.price_history) > 0:
            last_price = self.price_history[-1]
            if last_price > 0:
                change_percent = abs(price - last_price) / last_price * 100
                if change_percent > self.max_change_percent:
                    return ValidationResult.FAIL, f"Price change {change_percent:.2f}% exceeds maximum {self.max_change_percent}%"
        
        # Update price history
        self.price_history.append(price)
        
        return ValidationResult.PASS, None

class VolumeValidationRule(ValidationRule):
    """Specialized validation for volume data"""
    
    def __init__(self, name: str, min_volume: int = 0, max_volume: int = 1000000,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.min_volume = min_volume
        self.max_volume = max_volume
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, (int, float, np.number)):
            return ValidationResult.FAIL, f"Expected numeric volume, got {type(data)}"
        
        volume = int(data)
        
        if volume < self.min_volume:
            return ValidationResult.FAIL, f"Volume {volume} below minimum {self.min_volume}"
        
        if volume > self.max_volume:
            return ValidationResult.FAIL, f"Volume {volume} above maximum {self.max_volume}"
        
        return ValidationResult.PASS, None

class TimestampValidationRule(ValidationRule):
    """Validate timestamp data"""
    
    def __init__(self, name: str, max_age_seconds: float = 60.0, 
                 future_tolerance_seconds: float = 5.0,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.max_age_seconds = max_age_seconds
        self.future_tolerance_seconds = future_tolerance_seconds
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, (int, float, np.number)):
            return ValidationResult.FAIL, f"Expected numeric timestamp, got {type(data)}"
        
        timestamp = float(data)
        current_time = time.time()
        
        # Check if timestamp is too old
        age = current_time - timestamp
        if age > self.max_age_seconds:
            return ValidationResult.FAIL, f"Timestamp {age:.2f}s old, exceeds maximum {self.max_age_seconds}s"
        
        # Check if timestamp is too far in the future
        if timestamp > current_time + self.future_tolerance_seconds:
            return ValidationResult.FAIL, f"Timestamp {timestamp - current_time:.2f}s in future, exceeds tolerance {self.future_tolerance_seconds}s"
        
        return ValidationResult.PASS, None

class SequenceValidationRule(ValidationRule):
    """Validate sequence order"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(name, severity)
        self.last_sequence = None
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, (int, np.integer)):
            return ValidationResult.FAIL, f"Expected integer sequence, got {type(data)}"
        
        sequence = int(data)
        
        if self.last_sequence is not None:
            if sequence <= self.last_sequence:
                return ValidationResult.FAIL, f"Sequence {sequence} not greater than previous {self.last_sequence}"
            
            # Check for large gaps
            gap = sequence - self.last_sequence
            if gap > 1000:  # Configurable threshold
                return ValidationResult.FAIL, f"Large sequence gap: {gap}"
        
        self.last_sequence = sequence
        return ValidationResult.PASS, None

class DataFrameValidationRule(ValidationRule):
    """Validate pandas DataFrame structure"""
    
    def __init__(self, name: str, required_columns: List[str], 
                 max_rows: int = 10000, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.required_columns = required_columns
        self.max_rows = max_rows
    
    def _validate_impl(self, data: Any, field: str = "", metadata: Optional[Dict[str, Any]] = None) -> Tuple[ValidationResult, Optional[str]]:
        if not isinstance(data, pd.DataFrame):
            return ValidationResult.FAIL, f"Expected DataFrame, got {type(data)}"
        
        # Check if DataFrame is empty
        if len(data) == 0:
            return ValidationResult.FAIL, "DataFrame is empty"
        
        # Check row count
        if len(data) > self.max_rows:
            return ValidationResult.FAIL, f"DataFrame has {len(data)} rows, exceeds maximum {self.max_rows}"
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            return ValidationResult.FAIL, f"Missing required columns: {missing_columns}"
        
        # Check for all NaN columns
        for col in data.columns:
            if data[col].isna().all():
                return ValidationResult.FAIL, f"Column '{col}' contains only NaN values"
        
        return ValidationResult.PASS, None

class RealtimeDataValidator:
    """High-performance real-time data validator"""
    
    def __init__(self, max_workers: int = 4, enable_parallel: bool = True):
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        
        # Validation rules by field
        self.field_rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self.global_rules: List[ValidationRule] = []
        
        # Metrics tracking
        self.metrics = ValidationMetrics()
        self.metrics_lock = threading.Lock()
        
        # Issue tracking
        self.recent_issues = deque(maxlen=10000)
        self.issue_counts = defaultdict(int)
        
        # Performance tracking
        self.validation_times = deque(maxlen=1000)
        
        # Thread pool for parallel validation
        if enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("RealtimeDataValidator cleanup completed")
    
    def add_field_rule(self, field: str, rule: ValidationRule):
        """Add validation rule for specific field"""
        self.field_rules[field].append(rule)
    
    def add_global_rule(self, rule: ValidationRule):
        """Add global validation rule"""
        self.global_rules.append(rule)
    
    def validate_record(self, record: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
        """Validate a single record"""
        start_time = time.time_ns()
        issues = []
        
        try:
            # Validate global rules
            for rule in self.global_rules:
                result, message = rule.validate(record, "", metadata)
                if result == ValidationResult.FAIL:
                    issue = ValidationIssue(
                        field="global",
                        message=message or "Global validation failed",
                        severity=rule.severity,
                        value=record,
                        rule_name=rule.name,
                        metadata=metadata
                    )
                    issues.append(issue)
                    self._track_issue(issue)
            
            # Validate field-specific rules
            for field, value in record.items():
                if field in self.field_rules:
                    for rule in self.field_rules[field]:
                        result, message = rule.validate(value, field, metadata)
                        if result == ValidationResult.FAIL:
                            issue = ValidationIssue(
                                field=field,
                                message=message or f"Validation failed for field {field}",
                                severity=rule.severity,
                                value=value,
                                rule_name=rule.name,
                                metadata=metadata
                            )
                            issues.append(issue)
                            self._track_issue(issue)
            
            # Update metrics
            end_time = time.time_ns()
            validation_time_us = (end_time - start_time) / 1000
            self._update_metrics(validation_time_us, len(issues))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating record: {str(e)}")
            issue = ValidationIssue(
                field="validation_system",
                message=f"Validation system error: {str(e)}",
                severity=ValidationSeverity.CRITICAL,
                value=record,
                rule_name="system",
                metadata=metadata
            )
            return [issue]
    
    def validate_batch(self, records: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[List[ValidationIssue]]:
        """Validate a batch of records"""
        if not self.enable_parallel or len(records) < 10:
            # Sequential validation for small batches
            return [self.validate_record(record, metadata) for record in records]
        else:
            # Parallel validation for large batches
            futures = [self.executor.submit(self.validate_record, record, metadata) for record in records]
            return [future.result() for future in futures]
    
    def validate_dataframe(self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> DataQualityReport:
        """Validate entire DataFrame"""
        start_time = time.time()
        all_issues = []
        field_stats = {}
        
        try:
            # Validate DataFrame structure
            df_issues = self.validate_record({'dataframe': df}, metadata)
            all_issues.extend(df_issues)
            
            # Calculate field statistics
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    field_stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'null_count': df[col].isna().sum(),
                        'null_percentage': (df[col].isna().sum() / len(df)) * 100
                    }
                else:
                    field_stats[col] = {
                        'unique_count': df[col].nunique(),
                        'null_count': df[col].isna().sum(),
                        'null_percentage': (df[col].isna().sum() / len(df)) * 100
                    }
            
            # Validate sample of records for performance
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            
            for _, row in sample_df.iterrows():
                record = row.to_dict()
                issues = self.validate_record(record, metadata)
                all_issues.extend(issues)
            
            # Calculate quality metrics
            total_records = len(df)
            invalid_records = len([issue for issue in all_issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
            valid_records = total_records - invalid_records
            quality_score = (valid_records / total_records) * 100 if total_records > 0 else 0
            
            # Create report
            report = DataQualityReport(
                timestamp=time.time(),
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=invalid_records,
                quality_score=quality_score,
                issues=all_issues,
                field_statistics=field_stats,
                validation_metrics=self.get_metrics()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return DataQualityReport(
                timestamp=time.time(),
                total_records=len(df) if df is not None else 0,
                valid_records=0,
                invalid_records=len(df) if df is not None else 0,
                quality_score=0.0,
                issues=[ValidationIssue(
                    field="validation_system",
                    message=f"DataFrame validation error: {str(e)}",
                    severity=ValidationSeverity.CRITICAL,
                    value=None,
                    rule_name="system"
                )],
                field_statistics={},
                validation_metrics=self.get_metrics()
            )
    
    def _track_issue(self, issue: ValidationIssue):
        """Track validation issue"""
        self.recent_issues.append(issue)
        self.issue_counts[issue.rule_name] += 1
    
    def _update_metrics(self, validation_time_us: float, issue_count: int):
        """Update validation metrics"""
        with self.metrics_lock:
            self.metrics.total_validations += 1
            if issue_count > 0:
                self.metrics.total_failures += 1
            
            # Update timing metrics
            self.validation_times.append(validation_time_us)
            if len(self.validation_times) > 0:
                self.metrics.avg_validation_time_us = statistics.mean(self.validation_times)
                self.metrics.max_validation_time_us = max(self.validation_times)
            
            # Update throughput
            current_time = time.time()
            if hasattr(self, '_last_throughput_update'):
                time_delta = current_time - self._last_throughput_update
                if time_delta >= 1.0:  # Update every second
                    self.metrics.throughput_validations_per_sec = len(self.validation_times) / time_delta
                    self._last_throughput_update = current_time
            else:
                self._last_throughput_update = current_time
    
    def get_metrics(self) -> ValidationMetrics:
        """Get current validation metrics"""
        with self.metrics_lock:
            return ValidationMetrics(
                total_validations=self.metrics.total_validations,
                total_failures=self.metrics.total_failures,
                total_warnings=self.metrics.total_warnings,
                avg_validation_time_us=self.metrics.avg_validation_time_us,
                max_validation_time_us=self.metrics.max_validation_time_us,
                throughput_validations_per_sec=self.metrics.throughput_validations_per_sec,
                timestamp=time.time()
            )
    
    def get_recent_issues(self, limit: int = 100) -> List[ValidationIssue]:
        """Get recent validation issues"""
        return list(self.recent_issues)[-limit:]
    
    def get_issue_summary(self) -> Dict[str, Any]:
        """Get summary of validation issues"""
        severity_counts = defaultdict(int)
        for issue in self.recent_issues:
            severity_counts[issue.severity.value] += 1
        
        return {
            'total_issues': len(self.recent_issues),
            'severity_counts': dict(severity_counts),
            'rule_issue_counts': dict(self.issue_counts),
            'top_failing_rules': sorted(self.issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_rule_performance(self) -> List[Dict[str, Any]]:
        """Get performance statistics for all rules"""
        performance_stats = []
        
        # Global rules
        for rule in self.global_rules:
            performance_stats.append(rule.get_performance_stats())
        
        # Field rules
        for field, rules in self.field_rules.items():
            for rule in rules:
                stats = rule.get_performance_stats()
                stats['field'] = field
                performance_stats.append(stats)
        
        return performance_stats
    
    def reset_metrics(self):
        """Reset all metrics and counters"""
        with self.metrics_lock:
            self.metrics = ValidationMetrics()
            self.recent_issues.clear()
            self.issue_counts.clear()
            self.validation_times.clear()
            
            # Reset rule counters
            for rule in self.global_rules:
                rule.call_count = 0
                rule.failure_count = 0
                rule.total_time_us = 0.0
            
            for rules in self.field_rules.values():
                for rule in rules:
                    rule.call_count = 0
                    rule.failure_count = 0
                    rule.total_time_us = 0.0

# Market data specific validator
class MarketDataValidator(RealtimeDataValidator):
    """Specialized validator for market data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_market_data_rules()
    
    def _setup_market_data_rules(self):
        """Setup standard market data validation rules"""
        # Price validation
        self.add_field_rule('price', NotNullValidationRule('price_not_null'))
        self.add_field_rule('price', TypeValidationRule('price_type', (int, float, np.number)))
        self.add_field_rule('price', PriceValidationRule('price_range', min_price=0.01, max_price=100000))
        
        # Volume validation
        self.add_field_rule('volume', NotNullValidationRule('volume_not_null'))
        self.add_field_rule('volume', TypeValidationRule('volume_type', (int, np.integer)))
        self.add_field_rule('volume', VolumeValidationRule('volume_range', min_volume=1, max_volume=1000000))
        
        # Timestamp validation
        self.add_field_rule('timestamp', NotNullValidationRule('timestamp_not_null'))
        self.add_field_rule('timestamp', TypeValidationRule('timestamp_type', (int, float, np.number)))
        self.add_field_rule('timestamp', TimestampValidationRule('timestamp_range', max_age_seconds=60))
        
        # Symbol validation
        self.add_field_rule('symbol', NotNullValidationRule('symbol_not_null'))
        self.add_field_rule('symbol', TypeValidationRule('symbol_type', str))
        
        # Sequence validation
        self.add_field_rule('sequence', SequenceValidationRule('sequence_order', ValidationSeverity.WARNING))

# Utility functions
def create_market_data_validator() -> MarketDataValidator:
    """Create a pre-configured market data validator"""
    return MarketDataValidator(max_workers=4, enable_parallel=True)

def create_basic_validator() -> RealtimeDataValidator:
    """Create a basic validator with common rules"""
    validator = RealtimeDataValidator(max_workers=2, enable_parallel=True)
    
    # Add common rules
    validator.add_global_rule(NotNullValidationRule('global_not_null', ValidationSeverity.WARNING))
    
    return validator

def validate_market_tick(symbol: str, price: float, volume: int, timestamp: float) -> List[ValidationIssue]:
    """Quick validation for market tick data"""
    validator = create_market_data_validator()
    
    record = {
        'symbol': symbol,
        'price': price,
        'volume': volume,
        'timestamp': timestamp
    }
    
    return validator.validate_record(record)
