"""
Tactical Environment - 5-Minute Matrix Processing

Environment for processing 60×7 matrices with FVG, momentum, and entry agents.
"""

import numpy as np
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import logging
import hashlib
import hmac
import time
from collections import defaultdict

from .exceptions import MatrixValidationError, CriticalDependencyError

logger = logging.getLogger(__name__)

class TacticalEnvironment:
    """
    Tactical trading environment for 5-minute matrix processing.
    
    Handles 60x7 matrix inputs with features:
    - fvg_bullish_active, fvg_bearish_active, fvg_nearest_level, fvg_age
    - fvg_mitigation_signal, price_momentum_5, volume_ratio
    """
    
    def __init__(self, matrix_integrity_key: Optional[str] = None):
        """Initialize tactical environment."""
        self.matrix_shape = (60, 7)
        self.feature_names = [
            'fvg_bullish_active',
            'fvg_bearish_active', 
            'fvg_nearest_level',
            'fvg_age',
            'fvg_mitigation_signal',
            'price_momentum_5',
            'volume_ratio'
        ]
        
        # Security configuration
        self.matrix_integrity_key = matrix_integrity_key or "tactical_default_key_change_in_prod"
        self.validation_cache = {}  # Cache for expensive validations
        self.adversarial_detection_threshold = 3.0  # Standard deviations for anomaly detection
        self.max_validation_cache_size = 1000
        
        # Feature validation ranges
        self.feature_ranges = {
            'fvg_bullish_active': (0.0, 1.0),     # Binary feature
            'fvg_bearish_active': (0.0, 1.0),     # Binary feature
            'fvg_nearest_level': (0.90, 1.10),    # Price levels (10% range)
            'fvg_age': (0.0, 100.0),              # Age in periods
            'fvg_mitigation_signal': (0.0, 1.0),  # Binary feature
            'price_momentum_5': (-20.0, 20.0),    # Momentum percentage
            'volume_ratio': (0.0, 10.0)           # Volume ratio
        }
        
        # Adversarial pattern detection history
        self.suspicious_patterns = defaultdict(int)
        self.last_pattern_reset = time.time()
        
        # Agent configurations
        self.agents = {
            'fvg_agent': {
                'feature_focus': [0, 1, 2, 3, 4],  # FVG features
                'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05, 0.0, 0.0]
            },
            'momentum_agent': {
                'feature_focus': [5, 6],  # Momentum and volume
                'attention_weights': [0.05, 0.05, 0.1, 0.0, 0.0, 0.5, 0.3]
            },
            'entry_agent': {
                'feature_focus': [0, 1, 2, 3, 4, 5, 6],  # All features
                'attention_weights': [0.15, 0.15, 0.15, 0.1, 0.1, 0.2, 0.15]
            }
        }
        
        logger.info("Tactical environment initialized")
    
    async def initialize(self):
        """Initialize environment components."""
        logger.info("Tactical environment components initialized")
    
    def validate_matrix(
        self, 
        matrix: np.ndarray, 
        correlation_id: Optional[str] = None,
        integrity_hash: Optional[str] = None,
        strict_validation: bool = True
    ) -> bool:
        """
        Comprehensive matrix validation with security hardening.
        
        Args:
            matrix: Input matrix to validate
            correlation_id: Request correlation ID for error tracking
            integrity_hash: HMAC hash for integrity verification
            strict_validation: Enable strict validation (recommended for production)
        
        Returns:
            bool: True if matrix passes all validation checks
            
        Raises:
            MatrixValidationError: When validation fails with detailed error information
        """
        failed_checks = []
        
        try:
            # 1. Basic shape validation
            if matrix.shape != self.matrix_shape:
                failed_checks.append("invalid_shape")
                raise MatrixValidationError(
                    validation_type="shape_check",
                    error_message=f"Invalid matrix shape: {matrix.shape}, expected {self.matrix_shape}",
                    matrix_shape=matrix.shape,
                    failed_checks=failed_checks,
                    correlation_id=correlation_id
                )
            
            # 2. Data type validation
            if not np.issubdtype(matrix.dtype, np.floating):
                failed_checks.append("invalid_dtype")
                raise MatrixValidationError(
                    validation_type="dtype_check",
                    error_message=f"Invalid matrix dtype: {matrix.dtype}, expected floating point",
                    matrix_shape=matrix.shape,
                    failed_checks=failed_checks,
                    correlation_id=correlation_id
                )
            
            # 3. NaN/Inf injection detection (CRITICAL SECURITY CHECK)
            if not np.all(np.isfinite(matrix)):
                nan_count = np.count_nonzero(~np.isfinite(matrix))
                failed_checks.append("nan_inf_injection")
                raise MatrixValidationError(
                    validation_type="nan_inf_check",
                    error_message=f"Matrix contains {nan_count} NaN/infinite values - potential injection attack",
                    matrix_shape=matrix.shape,
                    failed_checks=failed_checks,
                    correlation_id=correlation_id
                )
            
            # 4. Feature range validation (prevents out-of-bounds attacks)
            for feature_idx, feature_name in enumerate(self.feature_names):
                feature_data = matrix[:, feature_idx]
                min_val, max_val = self.feature_ranges[feature_name]
                
                if np.any(feature_data < min_val) or np.any(feature_data > max_val):
                    out_of_range_count = np.count_nonzero((feature_data < min_val) | (feature_data > max_val))
                    failed_checks.append(f"range_violation_{feature_name}")
                    raise MatrixValidationError(
                        validation_type="range_check",
                        error_message=(
                            f"Feature '{feature_name}' has {out_of_range_count} values outside "
                            f"valid range [{min_val}, {max_val}] - potential attack"
                        ),
                        matrix_shape=matrix.shape,
                        failed_checks=failed_checks,
                        correlation_id=correlation_id
                    )
            
            # 5. Binary feature strict validation
            binary_features = [0, 1, 4]  # fvg_bullish_active, fvg_bearish_active, fvg_mitigation_signal
            for feature_idx in binary_features:
                feature_data = matrix[:, feature_idx]
                unique_values = np.unique(feature_data)
                
                if not np.all(np.isin(unique_values, [0.0, 1.0])):
                    failed_checks.append(f"binary_violation_{self.feature_names[feature_idx]}")
                    raise MatrixValidationError(
                        validation_type="binary_check",
                        error_message=(
                            f"Binary feature '{self.feature_names[feature_idx]}' contains "
                            f"non-binary values: {unique_values} - expected only [0, 1]"
                        ),
                        matrix_shape=matrix.shape,
                        failed_checks=failed_checks,
                        correlation_id=correlation_id
                    )
            
            # 6. Adversarial pattern detection (strict mode only)
            if strict_validation:
                self._detect_adversarial_patterns(matrix, correlation_id, failed_checks)
            
            # 7. Cryptographic integrity verification (if hash provided)
            if integrity_hash:
                if not self._verify_matrix_integrity(matrix, integrity_hash):
                    failed_checks.append("integrity_verification_failed")
                    raise MatrixValidationError(
                        validation_type="integrity_check",
                        error_message="Matrix integrity verification failed - data may be tampered",
                        matrix_shape=matrix.shape,
                        failed_checks=failed_checks,
                        correlation_id=correlation_id
                    )
            
            # 8. Statistical anomaly detection
            if strict_validation:
                self._detect_statistical_anomalies(matrix, correlation_id, failed_checks)
            
            logger.info(
                f"Matrix validation passed for shape {matrix.shape}",
                extra={"correlation_id": correlation_id, "validation_level": "strict" if strict_validation else "basic"}
            )
            return True
            
        except MatrixValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            failed_checks.append("unexpected_error")
            raise MatrixValidationError(
                validation_type="unexpected_error",
                error_message=f"Unexpected validation error: {str(e)}",
                matrix_shape=matrix.shape if matrix is not None else None,
                failed_checks=failed_checks,
                correlation_id=correlation_id
            ) from e
    
    def preprocess_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Preprocess matrix for agent consumption."""
        if not self.validate_matrix(matrix):
            raise ValueError("Invalid matrix provided")
        
        # Normalize features
        processed = matrix.copy()
        
        # Normalize continuous features
        for i, feature_name in enumerate(self.feature_names):
            if feature_name not in ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_mitigation_signal']:
                # Apply feature-specific normalization
                if feature_name == 'fvg_nearest_level':
                    # Normalize as percentage from current price (simulated)
                    processed[:, i] = np.clip(processed[:, i], 0.98, 1.02)
                elif feature_name == 'fvg_age':
                    # Apply exponential decay
                    processed[:, i] = np.exp(-processed[:, i] * 0.1)
                elif feature_name == 'price_momentum_5':
                    # Clip to reasonable range
                    processed[:, i] = np.clip(processed[:, i], -10, 10) / 10
                elif feature_name == 'volume_ratio':
                    # Log transform and normalize
                    processed[:, i] = np.tanh(np.log1p(np.maximum(processed[:, i], 0.1)))
        
        return processed
    
    def _detect_adversarial_patterns(
        self, 
        matrix: np.ndarray, 
        correlation_id: Optional[str],
        failed_checks: List[str]
    ):
        """
        Detect potential adversarial patterns in matrix data.
        
        Looks for:
        - Unusual value distributions
        - Gradient-based attack patterns
        - Repeated suspicious patterns
        """
        try:
            # Check for gradient-like patterns (common in adversarial attacks)
            for feature_idx in range(matrix.shape[1]):
                feature_data = matrix[:, feature_idx]
                
                # Detect monotonic sequences (potential gradient attacks)
                diff = np.diff(feature_data)
                if len(diff) > 0:
                    # Check for suspiciously monotonic sequences
                    positive_streak = 0
                    negative_streak = 0
                    max_positive_streak = 0
                    max_negative_streak = 0
                    
                    for d in diff:
                        if d > 0:
                            positive_streak += 1
                            negative_streak = 0
                            max_positive_streak = max(max_positive_streak, positive_streak)
                        elif d < 0:
                            negative_streak += 1
                            positive_streak = 0
                            max_negative_streak = max(max_negative_streak, negative_streak)
                        else:
                            positive_streak = 0
                            negative_streak = 0
                    
                    # Flag suspicious monotonic sequences
                    if max_positive_streak > 20 or max_negative_streak > 20:
                        pattern_key = f"monotonic_{feature_idx}_{max_positive_streak}_{max_negative_streak}"
                        self.suspicious_patterns[pattern_key] += 1
                        
                        if self.suspicious_patterns[pattern_key] > 3:  # Repeated suspicious pattern
                            failed_checks.append(f"adversarial_monotonic_{self.feature_names[feature_idx]}")
                            raise MatrixValidationError(
                                validation_type="adversarial_detection",
                                error_message=(
                                    f"Potential gradient-based attack detected in '{self.feature_names[feature_idx]}': "
                                    f"monotonic sequence length {max(max_positive_streak, max_negative_streak)}"
                                ),
                                matrix_shape=matrix.shape,
                                failed_checks=failed_checks,
                                correlation_id=correlation_id
                            )
            
            # Reset pattern counters periodically
            current_time = time.time()
            if current_time - self.last_pattern_reset > 300:  # 5 minutes
                self.suspicious_patterns.clear()
                self.last_pattern_reset = current_time
                
        except MatrixValidationError:
            raise
        except Exception as e:
            logger.warning(f"Error in adversarial pattern detection: {e}")
    
    def _detect_statistical_anomalies(
        self,
        matrix: np.ndarray,
        correlation_id: Optional[str],
        failed_checks: List[str]
    ):
        """
        Detect statistical anomalies that might indicate attacks or corrupted data.
        """
        try:
            for feature_idx, feature_name in enumerate(self.feature_names):
                feature_data = matrix[:, feature_idx]
                
                # Skip binary features for statistical analysis
                if feature_idx in [0, 1, 4]:  # Binary features
                    continue
                
                # Calculate statistics
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                
                if std_val > 0:  # Avoid division by zero
                    # Check for outliers beyond threshold standard deviations
                    z_scores = np.abs((feature_data - mean_val) / std_val)
                    outlier_count = np.count_nonzero(z_scores > self.adversarial_detection_threshold)
                    
                    # Flag if too many outliers (potential attack)
                    outlier_percentage = outlier_count / len(feature_data)
                    if outlier_percentage > 0.1:  # More than 10% outliers
                        failed_checks.append(f"statistical_anomaly_{feature_name}")
                        raise MatrixValidationError(
                            validation_type="statistical_anomaly",
                            error_message=(
                                f"Statistical anomaly detected in '{feature_name}': "
                                f"{outlier_percentage:.1%} values are outliers (>{self.adversarial_detection_threshold}σ)"
                            ),
                            matrix_shape=matrix.shape,
                            failed_checks=failed_checks,
                            correlation_id=correlation_id
                        )
                        
        except MatrixValidationError:
            raise
        except Exception as e:
            logger.warning(f"Error in statistical anomaly detection: {e}")
    
    def _verify_matrix_integrity(self, matrix: np.ndarray, expected_hash: str) -> bool:
        """
        Verify matrix integrity using HMAC-SHA256.
        
        Args:
            matrix: Matrix data to verify
            expected_hash: Expected HMAC hash
            
        Returns:
            bool: True if integrity check passes
        """
        try:
            # Convert matrix to bytes for hashing
            matrix_bytes = matrix.tobytes()
            
            # Calculate HMAC-SHA256
            calculated_hash = hmac.new(
                self.matrix_integrity_key.encode(),
                matrix_bytes,
                hashlib.sha256
            ).hexdigest()
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(calculated_hash, expected_hash)
            
        except Exception as e:
            logger.error(f"Error verifying matrix integrity: {e}")
            return False
    
    def generate_matrix_integrity_hash(self, matrix: np.ndarray) -> str:
        """
        Generate integrity hash for a matrix.
        
        Args:
            matrix: Matrix to generate hash for
            
        Returns:
            str: HMAC-SHA256 hash
        """
        try:
            matrix_bytes = matrix.tobytes()
            return hmac.new(
                self.matrix_integrity_key.encode(),
                matrix_bytes,
                hashlib.sha256
            ).hexdigest()
        except Exception as e:
            logger.error(f"Error generating matrix integrity hash: {e}")
            raise CriticalDependencyError(
                dependency="matrix_integrity",
                error_message=f"Failed to generate integrity hash: {e}"
            )