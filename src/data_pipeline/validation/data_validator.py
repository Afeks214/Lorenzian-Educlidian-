"""
Data validation system for large datasets
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import hashlib

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataValidationException
from ..core.data_loader import DataChunk
from ..streaming.data_streamer import DataStreamer

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    chunk_id: Optional[int] = None
    row_indices: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'chunk_id': self.chunk_id,
            'row_indices': self.row_indices
        }

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    enable_statistical_validation: bool = True
    enable_schema_validation: bool = True
    enable_business_rule_validation: bool = True
    enable_data_quality_checks: bool = True
    
    # Thresholds
    null_threshold: float = 0.1  # 10% max null values
    duplicate_threshold: float = 0.05  # 5% max duplicates
    outlier_threshold: float = 0.01  # 1% max outliers
    
    # Performance
    sample_size: int = 1000  # Sample size for statistical validation
    validation_timeout: float = 300.0  # 5 minutes
    enable_parallel_validation: bool = True
    max_workers: int = 4
    
    # Reporting
    enable_detailed_reports: bool = True
    report_directory: str = "/tmp/validation_reports"
    enable_realtime_alerts: bool = True
    
    # Persistence
    enable_validation_history: bool = True
    history_retention_days: int = 30

class ValidationRule:
    """Base class for validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data and return results"""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Get rule description"""
        return f"Validation rule: {self.name}"

class SchemaValidationRule(ValidationRule):
    """Validate data schema"""
    
    def __init__(self, expected_columns: List[str], 
                 expected_dtypes: Optional[Dict[str, str]] = None,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__("schema_validation", severity)
        self.expected_columns = expected_columns
        self.expected_dtypes = expected_dtypes or {}
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate schema"""
        results = []
        
        # Check columns
        missing_columns = set(self.expected_columns) - set(data.columns)
        if missing_columns:
            results.append(ValidationResult(
                rule_name=self.name,
                severity=self.severity,
                message=f"Missing columns: {list(missing_columns)}",
                details={'missing_columns': list(missing_columns)}
            ))
        
        extra_columns = set(data.columns) - set(self.expected_columns)
        if extra_columns:
            results.append(ValidationResult(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Extra columns: {list(extra_columns)}",
                details={'extra_columns': list(extra_columns)}
            ))
        
        # Check data types
        for col, expected_dtype in self.expected_dtypes.items():
            if col in data.columns:
                if str(data[col].dtype) != expected_dtype:
                    results.append(ValidationResult(
                        rule_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column {col} has dtype {data[col].dtype}, expected {expected_dtype}",
                        details={
                            'column': col,
                            'actual_dtype': str(data[col].dtype),
                            'expected_dtype': expected_dtype
                        }
                    ))
        
        return results

class NullValidationRule(ValidationRule):
    """Validate null values"""
    
    def __init__(self, threshold: float = 0.1, 
                 severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__("null_validation", severity)
        self.threshold = threshold
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate null values"""
        results = []
        
        for column in data.columns:
            null_ratio = data[column].isnull().sum() / len(data)
            
            if null_ratio > self.threshold:
                results.append(ValidationResult(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Column {column} has {null_ratio:.2%} null values (threshold: {self.threshold:.2%})",
                    details={
                        'column': column,
                        'null_ratio': null_ratio,
                        'threshold': self.threshold,
                        'null_count': data[column].isnull().sum()
                    }
                ))
        
        return results

class DuplicateValidationRule(ValidationRule):
    """Validate duplicate values"""
    
    def __init__(self, threshold: float = 0.05,
                 severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__("duplicate_validation", severity)
        self.threshold = threshold
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate duplicates"""
        results = []
        
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / len(data)
        
        if duplicate_ratio > self.threshold:
            results.append(ValidationResult(
                rule_name=self.name,
                severity=self.severity,
                message=f"Dataset has {duplicate_ratio:.2%} duplicate rows (threshold: {self.threshold:.2%})",
                details={
                    'duplicate_ratio': duplicate_ratio,
                    'threshold': self.threshold,
                    'duplicate_count': duplicate_count
                }
            ))
        
        return results

class OutlierValidationRule(ValidationRule):
    """Validate outliers using statistical methods"""
    
    def __init__(self, threshold: float = 0.01, method: str = "iqr",
                 severity: ValidationSeverity = ValidationSeverity.INFO):
        super().__init__("outlier_validation", severity)
        self.threshold = threshold
        self.method = method
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate outliers"""
        results = []
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            outlier_indices = self._detect_outliers(data[column])
            outlier_ratio = len(outlier_indices) / len(data)
            
            if outlier_ratio > self.threshold:
                results.append(ValidationResult(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Column {column} has {outlier_ratio:.2%} outliers (threshold: {self.threshold:.2%})",
                    details={
                        'column': column,
                        'outlier_ratio': outlier_ratio,
                        'threshold': self.threshold,
                        'outlier_count': len(outlier_indices),
                        'method': self.method
                    },
                    row_indices=outlier_indices
                ))
        
        return results
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using specified method"""
        if self.method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers.index.tolist()
        
        elif self.method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3]
            return outliers.index.tolist()
        
        else:
            return []

class RangeValidationRule(ValidationRule):
    """Validate value ranges"""
    
    def __init__(self, column: str, min_val: Optional[float] = None, 
                 max_val: Optional[float] = None,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"range_validation_{column}", severity)
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate ranges"""
        results = []
        
        if self.column not in data.columns:
            return results
        
        series = data[self.column]
        
        # Check minimum value
        if self.min_val is not None:
            violations = series < self.min_val
            if violations.any():
                violation_indices = violations[violations].index.tolist()
                results.append(ValidationResult(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Column {self.column} has {violations.sum()} values below minimum {self.min_val}",
                    details={
                        'column': self.column,
                        'min_val': self.min_val,
                        'violation_count': violations.sum(),
                        'actual_min': series.min()
                    },
                    row_indices=violation_indices
                ))
        
        # Check maximum value
        if self.max_val is not None:
            violations = series > self.max_val
            if violations.any():
                violation_indices = violations[violations].index.tolist()
                results.append(ValidationResult(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Column {self.column} has {violations.sum()} values above maximum {self.max_val}",
                    details={
                        'column': self.column,
                        'max_val': self.max_val,
                        'violation_count': violations.sum(),
                        'actual_max': series.max()
                    },
                    row_indices=violation_indices
                ))
        
        return results

class DataValidator:
    """
    Comprehensive data validator for large datasets
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None,
                 pipeline_config: Optional[DataPipelineConfig] = None):
        self.config = config or ValidationConfig()
        self.pipeline_config = pipeline_config or DataPipelineConfig()
        
        # Initialize components
        self.data_streamer = DataStreamer(self.pipeline_config)
        self.rules: List[ValidationRule] = []
        self.validation_results: List[ValidationResult] = []
        self.stats = ValidationStats()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup reporting
        self._setup_reporting()
        
        # Setup persistence
        if self.config.enable_validation_history:
            self._setup_persistence()
    
    def _setup_reporting(self):
        """Setup reporting directory"""
        Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_persistence(self):
        """Setup validation history persistence"""
        db_path = Path(self.config.report_directory) / "validation_history.db"
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    timestamp REAL,
                    chunk_id INTEGER,
                    file_path TEXT,
                    result_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON validation_results(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rule_name 
                ON validation_results(rule_name)
            """)
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        with self._lock:
            self.rules.append(rule)
    
    def add_rules(self, rules: List[ValidationRule]):
        """Add multiple validation rules"""
        with self._lock:
            self.rules.extend(rules)
    
    def validate_chunk(self, chunk: DataChunk) -> List[ValidationResult]:
        """Validate a single data chunk"""
        try:
            start_time = time.time()
            results = []
            
            # Apply all validation rules
            for rule in self.rules:
                try:
                    rule_results = rule.validate(chunk.data)
                    for result in rule_results:
                        result.chunk_id = chunk.chunk_id
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error applying rule {rule.name}: {str(e)}")
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule execution failed: {str(e)}",
                        chunk_id=chunk.chunk_id
                    ))
            
            # Update statistics
            validation_time = time.time() - start_time
            self.stats.update_validation_time(validation_time)
            self.stats.chunks_validated += 1
            
            # Store results
            with self._lock:
                self.validation_results.extend(results)
            
            # Persist results if enabled
            if self.config.enable_validation_history:
                self._persist_results(results, chunk.file_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating chunk {chunk.chunk_id}: {str(e)}")
            raise DataValidationException(f"Validation failed for chunk {chunk.chunk_id}: {str(e)}")
    
    def validate_file(self, file_path: str, **kwargs) -> List[ValidationResult]:
        """Validate entire file"""
        all_results = []
        
        try:
            for chunk in self.data_streamer.stream_file(file_path, **kwargs):
                results = self.validate_chunk(chunk)
                all_results.extend(results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            raise DataValidationException(f"File validation failed: {str(e)}")
    
    def validate_multiple_files(self, file_paths: List[str], 
                              parallel: bool = True,
                              **kwargs) -> List[ValidationResult]:
        """Validate multiple files"""
        all_results = []
        
        if parallel and self.config.enable_parallel_validation:
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.validate_file, file_path, **kwargs)
                    for file_path in file_paths
                ]
                
                for future in futures:
                    try:
                        results = future.result(timeout=self.config.validation_timeout)
                        all_results.extend(results)
                    except Exception as e:
                        logger.error(f"Error in parallel validation: {str(e)}")
        else:
            for file_path in file_paths:
                results = self.validate_file(file_path, **kwargs)
                all_results.extend(results)
        
        return all_results
    
    def _persist_results(self, results: List[ValidationResult], file_path: str):
        """Persist validation results"""
        try:
            db_path = Path(self.config.report_directory) / "validation_history.db"
            
            with sqlite3.connect(str(db_path)) as conn:
                for result in results:
                    result_dict = result.to_dict()
                    result_hash = hashlib.sha256(
                        json.dumps(result_dict, sort_keys=True).encode()
                    ).hexdigest()
                    
                    conn.execute("""
                        INSERT INTO validation_results 
                        (rule_name, severity, message, details, timestamp, chunk_id, file_path, result_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.rule_name,
                        result.severity.value,
                        result.message,
                        json.dumps(result.details),
                        result.timestamp,
                        result.chunk_id,
                        file_path,
                        result_hash
                    ))
        except Exception as e:
            logger.error(f"Error persisting validation results: {str(e)}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        with self._lock:
            results_by_severity = defaultdict(int)
            results_by_rule = defaultdict(int)
            
            for result in self.validation_results:
                results_by_severity[result.severity.value] += 1
                results_by_rule[result.rule_name] += 1
            
            return {
                'total_results': len(self.validation_results),
                'results_by_severity': dict(results_by_severity),
                'results_by_rule': dict(results_by_rule),
                'chunks_validated': self.stats.chunks_validated,
                'avg_validation_time': self.stats.get_avg_validation_time(),
                'total_validation_time': self.stats.get_total_validation_time()
            }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate validation report"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = Path(self.config.report_directory) / f"validation_report_{timestamp}.json"
        
        # Generate comprehensive report
        report = {
            'timestamp': time.time(),
            'summary': self.get_validation_summary(),
            'rules': [
                {
                    'name': rule.name,
                    'severity': rule.severity.value,
                    'description': rule.get_description()
                }
                for rule in self.rules
            ],
            'results': [result.to_dict() for result in self.validation_results]
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report generated: {output_path}")
        return str(output_path)
    
    def get_failed_validations(self, severity: ValidationSeverity = ValidationSeverity.ERROR) -> List[ValidationResult]:
        """Get validations that failed with specified severity or higher"""
        severity_order = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 2,
            ValidationSeverity.CRITICAL: 3
        }
        
        min_severity_level = severity_order[severity]
        
        return [
            result for result in self.validation_results
            if severity_order[result.severity] >= min_severity_level
        ]
    
    def clear_results(self):
        """Clear validation results"""
        with self._lock:
            self.validation_results.clear()
            self.stats.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'chunks_validated': self.stats.chunks_validated,
            'total_results': len(self.validation_results),
            'avg_validation_time': self.stats.get_avg_validation_time(),
            'total_validation_time': self.stats.get_total_validation_time(),
            'throughput': self.stats.get_throughput()
        }


class ValidationStats:
    """Statistics for validation operations"""
    
    def __init__(self):
        self.chunks_validated = 0
        self.validation_times = []
        self.start_time = time.time()
    
    def update_validation_time(self, validation_time: float):
        """Update validation time statistics"""
        self.validation_times.append(validation_time)
        
        # Keep only recent times (last 100)
        if len(self.validation_times) > 100:
            self.validation_times = self.validation_times[-100:]
    
    def get_avg_validation_time(self) -> float:
        """Get average validation time"""
        return np.mean(self.validation_times) if self.validation_times else 0.0
    
    def get_total_validation_time(self) -> float:
        """Get total validation time"""
        return sum(self.validation_times)
    
    def get_throughput(self) -> float:
        """Get validation throughput"""
        elapsed = time.time() - self.start_time
        return self.chunks_validated / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.chunks_validated = 0
        self.validation_times = []
        self.start_time = time.time()


# Utility functions
def create_default_validation_rules() -> List[ValidationRule]:
    """Create default validation rules"""
    return [
        NullValidationRule(threshold=0.1),
        DuplicateValidationRule(threshold=0.05),
        OutlierValidationRule(threshold=0.01)
    ]

def create_financial_data_rules() -> List[ValidationRule]:
    """Create validation rules for financial data"""
    return [
        SchemaValidationRule(
            expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            expected_dtypes={
                'open': 'float64',
                'high': 'float64',
                'low': 'float64', 
                'close': 'float64',
                'volume': 'int64'
            }
        ),
        RangeValidationRule('open', min_val=0),
        RangeValidationRule('high', min_val=0),
        RangeValidationRule('low', min_val=0),
        RangeValidationRule('close', min_val=0),
        RangeValidationRule('volume', min_val=0),
        NullValidationRule(threshold=0.01),  # Stricter for financial data
        DuplicateValidationRule(threshold=0.001)  # Very strict for financial data
    ]

def create_time_series_rules() -> List[ValidationRule]:
    """Create validation rules for time series data"""
    return [
        SchemaValidationRule(
            expected_columns=['timestamp', 'value'],
            expected_dtypes={'value': 'float64'}
        ),
        NullValidationRule(threshold=0.05),
        OutlierValidationRule(threshold=0.02, method="zscore")
    ]