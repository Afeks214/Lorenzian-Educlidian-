"""
Model Validator for Risk Management

This module provides comprehensive model validation capabilities including
input validation, output validation, and model behavior validation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
from abc import ABC, abstractmethod
import warnings
import json
from pathlib import Path

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class ValidationSeverity(Enum):
    """Severity levels for validation results"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Status of validation"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ValidationCategory(Enum):
    """Categories of validation"""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_VALIDATION = "output_validation"
    BEHAVIORAL_VALIDATION = "behavioral_validation"
    STATISTICAL_VALIDATION = "statistical_validation"
    PERFORMANCE_VALIDATION = "performance_validation"
    STABILITY_VALIDATION = "stability_validation"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_id: str
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    validation_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def is_failure(self) -> bool:
        """Check if validation failed"""
        return self.status == ValidationStatus.FAILED
    
    def is_critical(self) -> bool:
        """Check if validation is critical"""
        return self.severity == ValidationSeverity.CRITICAL


@dataclass
class ValidationSummary:
    """Summary of validation results"""
    total_validations: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_failures: int
    execution_time: float
    timestamp: datetime
    model_id: str
    model_version: str
    results: List[ValidationResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_validations == 0:
            return 0.0
        return self.passed / self.total_validations
    
    @property
    def has_critical_failures(self) -> bool:
        """Check if there are critical failures"""
        return self.critical_failures > 0
    
    @property
    def overall_status(self) -> ValidationStatus:
        """Determine overall validation status"""
        if self.has_critical_failures:
            return ValidationStatus.FAILED
        elif self.failed > 0:
            return ValidationStatus.FAILED
        elif self.warnings > 0:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASSED


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        category: ValidationCategory,
        severity: ValidationSeverity,
        enabled: bool = True,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.category = category
        self.severity = severity
        self.enabled = enabled
        self.description = description
        self.parameters = parameters or {}
        self.execution_count = 0
        self.failure_count = 0
        self.last_execution = None
        self.last_result = None
    
    @abstractmethod
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate model against rule"""
        pass
    
    def execute(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Execute validation with timing and error handling"""
        if not self.enabled:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.severity,
                status=ValidationStatus.SKIPPED,
                message="Rule disabled",
                details={},
                timestamp=datetime.now(),
                execution_time=0.0
            )
        
        start_time = datetime.now()
        
        try:
            result = self.validate(model, data, context)
            
            # Update execution statistics
            self.execution_count += 1
            self.last_execution = datetime.now()
            self.last_result = result
            
            if result.is_failure():
                self.failure_count += 1
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error("Validation rule execution failed", rule_id=self.rule_id, error=str(e))
            
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAILED,
                message=f"Validation execution failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=execution_time
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation rule statistics"""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "failure_count": self.failure_count,
            "failure_rate": self.failure_count / self.execution_count if self.execution_count > 0 else 0.0,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_status": self.last_result.status.value if self.last_result else None
        }


class InputValidationRule(ValidationRule):
    """Base class for input validation rules"""
    
    def __init__(self, rule_id: str, rule_name: str, **kwargs):
        super().__init__(
            rule_id=rule_id,
            rule_name=rule_name,
            category=ValidationCategory.INPUT_VALIDATION,
            **kwargs
        )


class OutputValidationRule(ValidationRule):
    """Base class for output validation rules"""
    
    def __init__(self, rule_id: str, rule_name: str, **kwargs):
        super().__init__(
            rule_id=rule_id,
            rule_name=rule_name,
            category=ValidationCategory.OUTPUT_VALIDATION,
            **kwargs
        )


class BehavioralValidationRule(ValidationRule):
    """Base class for behavioral validation rules"""
    
    def __init__(self, rule_id: str, rule_name: str, **kwargs):
        super().__init__(
            rule_id=rule_id,
            rule_name=rule_name,
            category=ValidationCategory.BEHAVIORAL_VALIDATION,
            **kwargs
        )


# Specific Input Validation Rules

class InputShapeValidation(InputValidationRule):
    """Validate input data shape"""
    
    def __init__(self, expected_shape: Tuple[int, ...], **kwargs):
        super().__init__(
            rule_id="input_shape_validation",
            rule_name="Input Shape Validation",
            severity=ValidationSeverity.ERROR,
            **kwargs
        )
        self.expected_shape = expected_shape
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate input data shape"""
        start_time = datetime.now()
        
        try:
            if hasattr(data, 'shape'):
                actual_shape = data.shape
                
                if actual_shape != self.expected_shape:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Input shape mismatch: expected {self.expected_shape}, got {actual_shape}",
                        details={
                            "expected_shape": self.expected_shape,
                            "actual_shape": actual_shape
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message="Input shape validation passed",
                        details={"shape": actual_shape},
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Input data does not have shape attribute",
                    details={"data_type": type(data).__name__},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Input shape validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


class InputRangeValidation(InputValidationRule):
    """Validate input data range"""
    
    def __init__(self, min_value: float, max_value: float, **kwargs):
        super().__init__(
            rule_id="input_range_validation",
            rule_name="Input Range Validation",
            severity=ValidationSeverity.WARNING,
            **kwargs
        )
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate input data range"""
        start_time = datetime.now()
        
        try:
            if hasattr(data, 'min') and hasattr(data, 'max'):
                data_min = float(data.min())
                data_max = float(data.max())
                
                if data_min < self.min_value or data_max > self.max_value:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Input data out of range: [{data_min:.4f}, {data_max:.4f}] not in [{self.min_value}, {self.max_value}]",
                        details={
                            "expected_min": self.min_value,
                            "expected_max": self.max_value,
                            "actual_min": data_min,
                            "actual_max": data_max
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message="Input range validation passed",
                        details={
                            "data_min": data_min,
                            "data_max": data_max
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Input data does not support min/max operations",
                    details={"data_type": type(data).__name__},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Input range validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


class InputNaNValidation(InputValidationRule):
    """Validate input data for NaN values"""
    
    def __init__(self, **kwargs):
        super().__init__(
            rule_id="input_nan_validation",
            rule_name="Input NaN Validation",
            severity=ValidationSeverity.ERROR,
            **kwargs
        )
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate input data for NaN values"""
        start_time = datetime.now()
        
        try:
            if hasattr(data, 'isnan') or hasattr(data, 'isnull'):
                if hasattr(data, 'isnan'):
                    has_nan = data.isnan().any()
                    if hasattr(has_nan, 'any'):  # For pandas/numpy multi-dimensional
                        has_nan = has_nan.any()
                else:
                    has_nan = data.isnull().any()
                    if hasattr(has_nan, 'any'):
                        has_nan = has_nan.any()
                
                if has_nan:
                    nan_count = 0
                    total_count = 0
                    
                    if hasattr(data, 'isnan'):
                        nan_count = data.isnan().sum()
                        if hasattr(nan_count, 'sum'):
                            nan_count = nan_count.sum()
                    elif hasattr(data, 'isnull'):
                        nan_count = data.isnull().sum()
                        if hasattr(nan_count, 'sum'):
                            nan_count = nan_count.sum()
                    
                    if hasattr(data, 'size'):
                        total_count = data.size
                    elif hasattr(data, '__len__'):
                        total_count = len(data)
                    
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Input data contains NaN values: {nan_count}/{total_count}",
                        details={
                            "nan_count": int(nan_count),
                            "total_count": int(total_count),
                            "nan_percentage": float(nan_count) / float(total_count) * 100 if total_count > 0 else 0
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message="Input NaN validation passed",
                        details={"nan_count": 0},
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Input data does not support NaN checking",
                    details={"data_type": type(data).__name__},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Input NaN validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


# Specific Output Validation Rules

class OutputShapeValidation(OutputValidationRule):
    """Validate output data shape"""
    
    def __init__(self, expected_shape: Tuple[int, ...], **kwargs):
        super().__init__(
            rule_id="output_shape_validation",
            rule_name="Output Shape Validation",
            severity=ValidationSeverity.ERROR,
            **kwargs
        )
        self.expected_shape = expected_shape
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate output data shape"""
        start_time = datetime.now()
        
        try:
            # Get model output
            if hasattr(model, 'predict'):
                output = model.predict(data)
            elif hasattr(model, 'forward'):
                output = model.forward(data)
            elif callable(model):
                output = model(data)
            else:
                output = context.get('output') if context else None
            
            if output is None:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAILED,
                    message="Could not obtain model output",
                    details={},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            if hasattr(output, 'shape'):
                actual_shape = output.shape
                
                if actual_shape != self.expected_shape:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Output shape mismatch: expected {self.expected_shape}, got {actual_shape}",
                        details={
                            "expected_shape": self.expected_shape,
                            "actual_shape": actual_shape
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message="Output shape validation passed",
                        details={"shape": actual_shape},
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Output data does not have shape attribute",
                    details={"output_type": type(output).__name__},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Output shape validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


class OutputRangeValidation(OutputValidationRule):
    """Validate output data range"""
    
    def __init__(self, min_value: float, max_value: float, **kwargs):
        super().__init__(
            rule_id="output_range_validation",
            rule_name="Output Range Validation",
            severity=ValidationSeverity.WARNING,
            **kwargs
        )
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate output data range"""
        start_time = datetime.now()
        
        try:
            # Get model output
            if hasattr(model, 'predict'):
                output = model.predict(data)
            elif hasattr(model, 'forward'):
                output = model.forward(data)
            elif callable(model):
                output = model(data)
            else:
                output = context.get('output') if context else None
            
            if output is None:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    status=ValidationStatus.FAILED,
                    message="Could not obtain model output",
                    details={},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            if hasattr(output, 'min') and hasattr(output, 'max'):
                output_min = float(output.min())
                output_max = float(output.max())
                
                if output_min < self.min_value or output_max > self.max_value:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Output data out of range: [{output_min:.4f}, {output_max:.4f}] not in [{self.min_value}, {self.max_value}]",
                        details={
                            "expected_min": self.min_value,
                            "expected_max": self.max_value,
                            "actual_min": output_min,
                            "actual_max": output_max
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message="Output range validation passed",
                        details={
                            "output_min": output_min,
                            "output_max": output_max
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Output data does not support min/max operations",
                    details={"output_type": type(output).__name__},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Output range validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


class MonotonicityValidation(BehavioralValidationRule):
    """Validate model monotonicity behavior"""
    
    def __init__(self, feature_index: int, expected_direction: str = "increasing", **kwargs):
        super().__init__(
            rule_id="monotonicity_validation",
            rule_name="Monotonicity Validation",
            severity=ValidationSeverity.WARNING,
            **kwargs
        )
        self.feature_index = feature_index
        self.expected_direction = expected_direction  # "increasing" or "decreasing"
    
    def validate(self, model: Any, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate model monotonicity behavior"""
        start_time = datetime.now()
        
        try:
            # Create test data with varying feature values
            if hasattr(data, 'copy'):
                test_data = data.copy()
            else:
                test_data = np.copy(data)
            
            # Generate range of values for the feature
            if hasattr(test_data, 'shape') and len(test_data.shape) > 1:
                original_value = test_data[0, self.feature_index]
                test_values = np.linspace(original_value * 0.5, original_value * 1.5, 10)
                
                outputs = []
                for test_value in test_values:
                    test_data[0, self.feature_index] = test_value
                    
                    if hasattr(model, 'predict'):
                        output = model.predict(test_data)
                    elif hasattr(model, 'forward'):
                        output = model.forward(test_data)
                    elif callable(model):
                        output = model(test_data)
                    else:
                        raise ValueError("Model does not have predict/forward method or is not callable")
                    
                    if hasattr(output, 'flatten'):
                        outputs.append(float(output.flatten()[0]))
                    else:
                        outputs.append(float(output))
                
                # Check monotonicity
                if self.expected_direction == "increasing":
                    is_monotonic = all(outputs[i] <= outputs[i+1] for i in range(len(outputs)-1))
                else:
                    is_monotonic = all(outputs[i] >= outputs[i+1] for i in range(len(outputs)-1))
                
                if not is_monotonic:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.FAILED,
                        message=f"Model violates {self.expected_direction} monotonicity for feature {self.feature_index}",
                        details={
                            "feature_index": self.feature_index,
                            "expected_direction": self.expected_direction,
                            "test_values": test_values.tolist(),
                            "outputs": outputs
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                else:
                    return ValidationResult(
                        rule_id=self.rule_id,
                        rule_name=self.rule_name,
                        category=self.category,
                        severity=self.severity,
                        status=ValidationStatus.PASSED,
                        message=f"Model satisfies {self.expected_direction} monotonicity for feature {self.feature_index}",
                        details={
                            "feature_index": self.feature_index,
                            "expected_direction": self.expected_direction,
                            "test_values": test_values.tolist(),
                            "outputs": outputs
                        },
                        timestamp=datetime.now(),
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            else:
                return ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    status=ValidationStatus.WARNING,
                    message="Data does not support monotonicity testing",
                    details={"data_shape": getattr(data, 'shape', 'unknown')},
                    timestamp=datetime.now(),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=ValidationSeverity.ERROR,
                status=ValidationStatus.FAILED,
                message=f"Monotonicity validation failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds()
            )


class ModelValidator:
    """Main model validation system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.validation_history: List[ValidationSummary] = []
        self.enabled = True
        
        # Statistics
        self.total_validations = 0
        self.total_failures = 0
        self.total_execution_time = 0.0
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Model Validator initialized")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        
        # Input validation rules
        self.add_rule(InputNaNValidation())
        self.add_rule(InputRangeValidation(min_value=-10.0, max_value=10.0))
        
        # Output validation rules
        self.add_rule(OutputRangeValidation(min_value=-1.0, max_value=1.0))
        
        # Behavioral validation rules
        # Note: These would be configured based on specific model requirements
        
    def add_rule(self, rule: ValidationRule) -> bool:
        """Add a validation rule"""
        try:
            self.validation_rules[rule.rule_id] = rule
            logger.info("Validation rule added", rule_id=rule.rule_id)
            return True
        except Exception as e:
            logger.error("Failed to add validation rule", rule_id=rule.rule_id, error=str(e))
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule"""
        try:
            if rule_id in self.validation_rules:
                del self.validation_rules[rule_id]
                logger.info("Validation rule removed", rule_id=rule_id)
                return True
            return False
        except Exception as e:
            logger.error("Failed to remove validation rule", rule_id=rule_id, error=str(e))
            return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = False
            return True
        return False
    
    def validate_model(
        self,
        model: Any,
        data: Any,
        model_id: str,
        model_version: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationSummary:
        """Validate model against all rules"""
        if not self.enabled:
            return ValidationSummary(
                total_validations=0,
                passed=0,
                failed=0,
                warnings=0,
                skipped=0,
                critical_failures=0,
                execution_time=0.0,
                timestamp=datetime.now(),
                model_id=model_id,
                model_version=model_version,
                results=[]
            )
        
        start_time = datetime.now()
        results = []
        
        # Execute all validation rules
        for rule_id, rule in self.validation_rules.items():
            try:
                result = rule.execute(model, data, context)
                result.model_id = model_id
                result.model_version = model_version
                results.append(result)
                
            except Exception as e:
                logger.error("Validation rule execution failed", rule_id=rule_id, error=str(e))
                
                # Create failure result
                failure_result = ValidationResult(
                    rule_id=rule_id,
                    rule_name=rule.rule_name,
                    category=rule.category,
                    severity=ValidationSeverity.CRITICAL,
                    status=ValidationStatus.FAILED,
                    message=f"Rule execution failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    execution_time=0.0,
                    model_id=model_id,
                    model_version=model_version
                )
                results.append(failure_result)
        
        # Calculate summary statistics
        total_validations = len(results)
        passed = len([r for r in results if r.status == ValidationStatus.PASSED])
        failed = len([r for r in results if r.status == ValidationStatus.FAILED])
        warnings = len([r for r in results if r.status == ValidationStatus.WARNING])
        skipped = len([r for r in results if r.status == ValidationStatus.SKIPPED])
        critical_failures = len([r for r in results if r.is_critical() and r.is_failure()])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create summary
        summary = ValidationSummary(
            total_validations=total_validations,
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            critical_failures=critical_failures,
            execution_time=execution_time,
            timestamp=datetime.now(),
            model_id=model_id,
            model_version=model_version,
            results=results
        )
        
        # Update statistics
        self.total_validations += total_validations
        self.total_failures += failed
        self.total_execution_time += execution_time
        
        # Store validation summary
        self.validation_history.append(summary)
        
        # Publish validation event
        self._publish_validation_event(summary)
        
        logger.info(
            "Model validation completed",
            model_id=model_id,
            model_version=model_version,
            overall_status=summary.overall_status.value,
            passed=passed,
            failed=failed,
            warnings=warnings,
            critical_failures=critical_failures,
            execution_time=execution_time
        )
        
        return summary
    
    def _publish_validation_event(self, summary: ValidationSummary):
        """Publish validation event to event bus"""
        try:
            event = self.event_bus.create_event(
                event_type=EventType.MODEL_VALIDATION,  # Assuming this exists
                payload={
                    "model_id": summary.model_id,
                    "model_version": summary.model_version,
                    "overall_status": summary.overall_status.value,
                    "total_validations": summary.total_validations,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "warnings": summary.warnings,
                    "critical_failures": summary.critical_failures,
                    "success_rate": summary.success_rate,
                    "execution_time": summary.execution_time,
                    "timestamp": summary.timestamp.isoformat()
                },
                source="model_validator"
            )
            
            self.event_bus.publish(event)
            
        except Exception as e:
            logger.error("Failed to publish validation event", error=str(e))
    
    def get_validation_history(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ValidationSummary]:
        """Get validation history"""
        history = self.validation_history.copy()
        
        if model_id:
            history = [h for h in history if h.model_id == model_id]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_rule_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all validation rules"""
        return {
            rule_id: rule.get_statistics()
            for rule_id, rule in self.validation_rules.items()
        }
    
    def get_validator_status(self) -> Dict[str, Any]:
        """Get validator status"""
        return {
            "enabled": self.enabled,
            "total_rules": len(self.validation_rules),
            "active_rules": len([r for r in self.validation_rules.values() if r.enabled]),
            "total_validations": self.total_validations,
            "total_failures": self.total_failures,
            "overall_failure_rate": self.total_failures / self.total_validations if self.total_validations > 0 else 0.0,
            "average_execution_time": self.total_execution_time / len(self.validation_history) if self.validation_history else 0.0,
            "validation_history_count": len(self.validation_history)
        }
    
    def generate_validation_report(
        self,
        model_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        try:
            # Filter history by model and period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            history = [
                h for h in self.validation_history
                if h.timestamp >= cutoff_date
                and (model_id is None or h.model_id == model_id)
            ]
            
            if not history:
                return {
                    "period_days": period_days,
                    "model_id": model_id,
                    "total_validations": 0,
                    "message": "No validation history found for the specified period"
                }
            
            # Calculate aggregate statistics
            total_validations = sum(h.total_validations for h in history)
            total_passed = sum(h.passed for h in history)
            total_failed = sum(h.failed for h in history)
            total_warnings = sum(h.warnings for h in history)
            total_critical = sum(h.critical_failures for h in history)
            
            # Calculate trends
            if len(history) >= 2:
                recent_success_rate = sum(h.success_rate for h in history[:5]) / min(5, len(history))
                older_success_rate = sum(h.success_rate for h in history[-5:]) / min(5, len(history))
                trend = "improving" if recent_success_rate > older_success_rate else "declining" if recent_success_rate < older_success_rate else "stable"
            else:
                trend = "insufficient_data"
            
            # Rule failure analysis
            rule_failures = {}
            for h in history:
                for result in h.results:
                    if result.is_failure():
                        rule_failures[result.rule_id] = rule_failures.get(result.rule_id, 0) + 1
            
            # Most problematic rules
            most_problematic = sorted(rule_failures.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report = {
                "period_days": period_days,
                "model_id": model_id,
                "report_generated": datetime.now().isoformat(),
                "summary": {
                    "total_validation_runs": len(history),
                    "total_validations": total_validations,
                    "total_passed": total_passed,
                    "total_failed": total_failed,
                    "total_warnings": total_warnings,
                    "total_critical": total_critical,
                    "overall_success_rate": total_passed / total_validations if total_validations > 0 else 0.0,
                    "trend": trend
                },
                "most_problematic_rules": [
                    {
                        "rule_id": rule_id,
                        "rule_name": self.validation_rules[rule_id].rule_name if rule_id in self.validation_rules else "Unknown",
                        "failure_count": count,
                        "failure_rate": count / total_validations if total_validations > 0 else 0.0
                    }
                    for rule_id, count in most_problematic
                ],
                "recent_validations": [
                    {
                        "timestamp": h.timestamp.isoformat(),
                        "model_id": h.model_id,
                        "model_version": h.model_version,
                        "overall_status": h.overall_status.value,
                        "success_rate": h.success_rate,
                        "critical_failures": h.critical_failures
                    }
                    for h in history[:10]  # Last 10 validations
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate validation report", error=str(e))
            return {"error": str(e)}
    
    def enable_validator(self):
        """Enable the validator"""
        self.enabled = True
        logger.info("Model validator enabled")
    
    def disable_validator(self):
        """Disable the validator"""
        self.enabled = False
        logger.info("Model validator disabled")
    
    def cleanup_history(self, days_old: int = 90):
        """Clean up old validation history"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        initial_count = len(self.validation_history)
        self.validation_history = [
            h for h in self.validation_history
            if h.timestamp > cutoff_date
        ]
        
        cleaned_count = initial_count - len(self.validation_history)
        
        if cleaned_count > 0:
            logger.info("Cleaned up old validation history", count=cleaned_count)
        
        return cleaned_count