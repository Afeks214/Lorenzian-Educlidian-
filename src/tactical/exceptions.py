"""
Tactical Trading System Exceptions

Custom exceptions for tactical trading system with fail-fast security patterns.
"""

import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CriticalDependencyError(Exception):
    """
    Critical dependency failure exception.
    
    This exception is raised when a critical dependency (like matrix data)
    fails and the system must fail-fast rather than proceed with invalid/random data.
    
    This prevents silent failures that could lead to:
    - Trading on invalid data
    - Financial losses due to corrupted inputs
    - Security vulnerabilities from malformed data
    """
    
    def __init__(
        self,
        dependency: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize critical dependency error.
        
        Args:
            dependency: Name of the failed dependency (e.g., "matrix_assembler", "redis_client")
            error_message: Human-readable error description
            error_details: Additional error context for debugging
            correlation_id: Request correlation ID for tracking
        """
        self.dependency = dependency
        self.error_message = error_message
        self.error_details = error_details or {}
        self.correlation_id = correlation_id
        self.timestamp = time.time()
        
        # Format error message
        formatted_message = f"CRITICAL DEPENDENCY FAILURE [{dependency}]: {error_message}"
        if correlation_id:
            formatted_message += f" [correlation_id: {correlation_id}]"
        
        super().__init__(formatted_message)
        
        # Log critical error immediately
        logger.critical(
            f"Critical dependency failure: {dependency}",
            extra={
                "dependency": dependency,
                "error_message": error_message,
                "error_details": error_details,
                "correlation_id": correlation_id,
                "timestamp": self.timestamp
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "exception_type": "CriticalDependencyError",
            "dependency": self.dependency,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


class MatrixValidationError(Exception):
    """
    Matrix validation failure exception.
    
    Raised when matrix input validation fails due to:
    - Invalid shape or dimensions
    - NaN/Inf injection attacks
    - Out-of-range values indicating potential attacks
    - Adversarial patterns detected
    - Cryptographic integrity failures
    """
    
    def __init__(
        self,
        validation_type: str,
        error_message: str,
        matrix_shape: Optional[tuple] = None,
        failed_checks: Optional[list] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize matrix validation error.
        
        Args:
            validation_type: Type of validation that failed (e.g., "shape", "nan_check", "range_check")
            error_message: Detailed error description
            matrix_shape: Shape of the matrix that failed validation
            failed_checks: List of specific validation checks that failed
            correlation_id: Request correlation ID for tracking
        """
        self.validation_type = validation_type
        self.error_message = error_message
        self.matrix_shape = matrix_shape
        self.failed_checks = failed_checks or []
        self.correlation_id = correlation_id
        self.timestamp = time.time()
        
        # Format error message
        formatted_message = f"MATRIX VALIDATION FAILURE [{validation_type}]: {error_message}"
        if matrix_shape:
            formatted_message += f" [shape: {matrix_shape}]"
        if correlation_id:
            formatted_message += f" [correlation_id: {correlation_id}]"
        
        super().__init__(formatted_message)
        
        # Log security-related validation failure
        logger.warning(
            f"Matrix validation failure: {validation_type}",
            extra={
                "validation_type": validation_type,
                "error_message": error_message,
                "matrix_shape": matrix_shape,
                "failed_checks": failed_checks,
                "correlation_id": correlation_id,
                "timestamp": self.timestamp
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "exception_type": "MatrixValidationError",
            "validation_type": self.validation_type,
            "error_message": self.error_message,
            "matrix_shape": self.matrix_shape,
            "failed_checks": self.failed_checks,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


class CircuitBreakerOpenError(Exception):
    """
    Circuit breaker is open, preventing execution.
    
    Raised when the circuit breaker pattern is activated due to:
    - Repeated dependency failures
    - System instability detected
    - Risk management protocols triggered
    """
    
    def __init__(
        self,
        circuit_name: str,
        failure_count: int,
        failure_threshold: int,
        time_window: float,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize circuit breaker error.
        
        Args:
            circuit_name: Name of the circuit breaker
            failure_count: Current failure count
            failure_threshold: Threshold that triggered the circuit breaker
            time_window: Time window for failure counting
            correlation_id: Request correlation ID for tracking
        """
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        self.failure_threshold = failure_threshold
        self.time_window = time_window
        self.correlation_id = correlation_id
        self.timestamp = time.time()
        
        formatted_message = (
            f"CIRCUIT BREAKER OPEN [{circuit_name}]: "
            f"{failure_count}/{failure_threshold} failures in {time_window}s window"
        )
        if correlation_id:
            formatted_message += f" [correlation_id: {correlation_id}]"
        
        super().__init__(formatted_message)
        
        logger.error(
            f"Circuit breaker opened: {circuit_name}",
            extra={
                "circuit_name": circuit_name,
                "failure_count": failure_count,
                "failure_threshold": failure_threshold,
                "time_window": time_window,
                "correlation_id": correlation_id,
                "timestamp": self.timestamp
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "exception_type": "CircuitBreakerOpenError",
            "circuit_name": self.circuit_name,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "time_window": self.time_window,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }