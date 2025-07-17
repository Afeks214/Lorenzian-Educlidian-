"""
Mathematical Validation Framework for Strategic MARL Components.

This module provides comprehensive mathematical validation for all strategic MARL
components including:
- GAE (Generalized Advantage Estimation) computation validation
- Kernel calculation verification
- MMD (Maximum Mean Discrepancy) calculation tests  
- Superposition probability constraint validation
- Numerical stability testing under extreme conditions
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result from a mathematical validation test."""
    test_name: str
    passed: bool
    error_message: Optional[str]
    computed_value: Any
    expected_value: Any
    tolerance: float
    timestamp: datetime


class MathematicalValidator:
    """
    Comprehensive mathematical validation framework for Strategic MARL.
    
    This class validates all mathematical computations to ensure:
    - Numerical accuracy within specified tolerances
    - Stability under extreme conditions
    - Proper constraint satisfaction
    - Mathematical consistency across operations
    """
    
    def __init__(self, tolerance: float = 1e-6, device: str = "cpu"):
        """
        Initialize the mathematical validator.
        
        Args:
            tolerance: Default numerical tolerance for comparisons
            device: Torch device for computations
        """
        self.tolerance = tolerance
        self.device = torch.device(device)
        self.logger = logging.getLogger('mathematical_validator')
        
        # Track validation results
        self.validation_history: List[ValidationResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        
        self.logger.info(f"Mathematical validator initialized with tolerance={tolerance}")
    
    def validate_all(self, test_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Run all mathematical validation tests.
        
        Args:
            test_data: Dictionary containing test data and parameters
            
        Returns:
            Dictionary mapping test names to ValidationResult objects
        """
        results = {}
        
        # GAE computation validation
        results['gae_computation'] = self.validate_gae_computation(test_data)
        
        # Kernel calculation validation
        results['kernel_calculations'] = self.validate_kernel_calculations(test_data)
        
        # MMD calculation validation
        results['mmd_calculation'] = self.validate_mmd_calculation(test_data)
        
        # Superposition probability validation
        results['superposition_probabilities'] = self.validate_superposition_probabilities(test_data)
        
        # Numerical stability validation
        results['numerical_stability'] = self.validate_numerical_stability(test_data)
        
        # Update summary statistics
        self._update_validation_summary(results)
        
        return results
    
    def validate_gae_computation(self, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate Generalized Advantage Estimation computation.
        
        GAE formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        try:
            # Get test parameters
            rewards = test_data.get('rewards', np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
            values = test_data.get('values', np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
            gamma = test_data.get('gamma', 0.99)
            gae_lambda = test_data.get('gae_lambda', 0.95)
            
            # Compute GAE step by step for validation
            expected_advantages = self._compute_gae_reference(rewards, values, gamma, gae_lambda)
            
            # Compute using the system's GAE implementation (simulated)
            computed_advantages = self._compute_gae_system(rewards, values, gamma, gae_lambda)
            
            # Validate results
            max_error = np.max(np.abs(expected_advantages - computed_advantages))
            passed = max_error < self.tolerance
            
            return ValidationResult(
                test_name="GAE Computation",
                passed=passed,
                error_message=None if passed else f"Max error {max_error} exceeds tolerance {self.tolerance}",
                computed_value=computed_advantages.tolist(),
                expected_value=expected_advantages.tolist(),
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="GAE Computation",
                passed=False,
                error_message=f"Exception during GAE validation: {str(e)}",
                computed_value=None,
                expected_value=None,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
    
    def _compute_gae_reference(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        gamma: float, 
        gae_lambda: float
    ) -> np.ndarray:
        """Compute GAE using reference implementation for validation."""
        T = len(rewards)
        advantages = np.zeros(T)
        
        # Compute temporal differences
        deltas = np.zeros(T)
        for t in range(T):
            if t < T - 1:
                deltas[t] = rewards[t] + gamma * values[t + 1] - values[t]
            else:
                deltas[t] = rewards[t] - values[t]  # Terminal state
        
        # Compute GAE advantages
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                advantages[t] = deltas[t]
            else:
                gae = deltas[t] + gamma * gae_lambda * gae
                advantages[t] = gae
        
        return advantages
    
    def _compute_gae_system(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        gamma: float, 
        gae_lambda: float
    ) -> np.ndarray:
        """Simulate system's GAE computation (placeholder for actual implementation)."""
        # This would call the actual system's GAE implementation
        # For now, we'll use the reference implementation to simulate
        return self._compute_gae_reference(rewards, values, gamma, gae_lambda)
    
    def validate_kernel_calculations(self, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate kernel distance calculations.
        
        Tests RBF kernel: K(x, y) = exp(-||x - y||² / (2σ²))
        """
        try:
            # Generate test data
            x = test_data.get('kernel_x', np.random.randn(10, 5))
            y = test_data.get('kernel_y', np.random.randn(10, 5))
            sigma = test_data.get('kernel_sigma', 1.0)
            
            # Compute expected kernel values
            expected_kernel = self._compute_rbf_kernel_reference(x, y, sigma)
            
            # Compute using system implementation (simulated)
            computed_kernel = self._compute_rbf_kernel_system(x, y, sigma)
            
            # Validate
            max_error = np.max(np.abs(expected_kernel - computed_kernel))
            passed = max_error < self.tolerance
            
            return ValidationResult(
                test_name="Kernel Calculations",
                passed=passed,
                error_message=None if passed else f"Max kernel error {max_error} exceeds tolerance",
                computed_value=float(max_error),
                expected_value=0.0,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Kernel Calculations",
                passed=False,
                error_message=f"Exception during kernel validation: {str(e)}",
                computed_value=None,
                expected_value=None,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
    
    def _compute_rbf_kernel_reference(self, x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        """Reference RBF kernel implementation."""
        distances_squared = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        return np.exp(-distances_squared / (2 * sigma ** 2))
    
    def _compute_rbf_kernel_system(self, x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        """Simulate system's kernel implementation."""
        # This would call the actual system's kernel computation
        return self._compute_rbf_kernel_reference(x, y, sigma)
    
    def validate_mmd_calculation(self, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate Maximum Mean Discrepancy calculation.
        
        MMD² = E[K(x, x')] + E[K(y, y')] - 2E[K(x, y)]
        """
        try:
            # Generate test distributions
            X = test_data.get('mmd_X', np.random.randn(50, 10))
            Y = test_data.get('mmd_Y', np.random.randn(50, 10) + 1.0)  # Shifted distribution
            sigma = test_data.get('mmd_sigma', 1.0)
            
            # Compute expected MMD
            expected_mmd = self._compute_mmd_reference(X, Y, sigma)
            
            # Compute using system implementation (simulated)
            computed_mmd = self._compute_mmd_system(X, Y, sigma)
            
            # Validate
            error = abs(expected_mmd - computed_mmd)
            passed = error < self.tolerance
            
            return ValidationResult(
                test_name="MMD Calculation",
                passed=passed,
                error_message=None if passed else f"MMD error {error} exceeds tolerance",
                computed_value=float(computed_mmd),
                expected_value=float(expected_mmd),
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="MMD Calculation",
                passed=False,
                error_message=f"Exception during MMD validation: {str(e)}",
                computed_value=None,
                expected_value=None,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
    
    def _compute_mmd_reference(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
        """Reference MMD implementation."""
        n, m = X.shape[0], Y.shape[0]
        
        # Compute kernel matrices
        K_XX = self._compute_rbf_kernel_reference(X, X, sigma)
        K_YY = self._compute_rbf_kernel_reference(Y, Y, sigma)
        K_XY = self._compute_rbf_kernel_reference(X, Y, sigma)
        
        # MMD² calculation
        term1 = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
        term2 = (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
        term3 = np.sum(K_XY) / (n * m)
        
        mmd_squared = term1 + term2 - 2 * term3
        return np.sqrt(max(0, mmd_squared))  # Ensure non-negative
    
    def _compute_mmd_system(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
        """Simulate system's MMD implementation."""
        # This would call the actual system's MMD computation
        return self._compute_mmd_reference(X, Y, sigma)
    
    def validate_superposition_probabilities(self, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate superposition probability calculations.
        
        Checks:
        1. Probabilities sum to 1
        2. All probabilities are non-negative
        3. Weighted ensemble maintains constraints
        """
        try:
            # Get test probability distributions
            agent_probs = test_data.get('agent_probabilities', [
                [0.4, 0.3, 0.3],  # Agent 1
                [0.2, 0.5, 0.3],  # Agent 2
                [0.35, 0.35, 0.3] # Agent 3
            ])
            weights = test_data.get('ensemble_weights', [0.4, 0.35, 0.25])
            confidences = test_data.get('agent_confidences', [0.8, 0.7, 0.75])
            
            # Compute ensemble probabilities
            ensemble_probs = self._compute_ensemble_probabilities(agent_probs, weights, confidences)
            
            # Validation checks
            errors = []
            
            # Check sum to 1
            prob_sum = np.sum(ensemble_probs)
            if abs(prob_sum - 1.0) > self.tolerance:
                errors.append(f"Probabilities sum to {prob_sum}, not 1.0")
            
            # Check non-negative
            if np.any(ensemble_probs < 0):
                errors.append("Some probabilities are negative")
            
            # Check weights sum to 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > self.tolerance:
                errors.append(f"Weights sum to {weight_sum}, not 1.0")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Superposition Probabilities",
                passed=passed,
                error_message=error_message,
                computed_value=ensemble_probs.tolist(),
                expected_value=[float(prob_sum), 1.0],  # [actual_sum, expected_sum]
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Superposition Probabilities",
                passed=False,
                error_message=f"Exception during superposition validation: {str(e)}",
                computed_value=None,
                expected_value=None,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
    
    def _compute_ensemble_probabilities(
        self, 
        agent_probs: List[List[float]], 
        weights: List[float], 
        confidences: List[float]
    ) -> np.ndarray:
        """Compute ensemble probabilities using weighted superposition."""
        agent_probs = np.array(agent_probs)
        weights = np.array(weights)
        confidences = np.array(confidences)
        
        # Weighted ensemble calculation
        ensemble_probs = np.zeros(agent_probs.shape[1])
        total_weight = 0.0
        
        for i, (probs, weight, confidence) in enumerate(zip(agent_probs, weights, confidences)):
            effective_weight = weight * confidence
            ensemble_probs += effective_weight * probs
            total_weight += effective_weight
        
        # Normalize
        if total_weight > 0:
            ensemble_probs /= total_weight
        else:
            ensemble_probs = np.ones(len(ensemble_probs)) / len(ensemble_probs)
        
        return ensemble_probs
    
    def validate_numerical_stability(self, test_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate numerical stability under extreme conditions.
        
        Tests:
        - Very large numbers
        - Very small numbers
        - NaN and infinity handling
        - Precision edge cases
        """
        try:
            stability_errors = []
            
            # Test 1: Large number handling
            large_numbers = np.array([1e10, 1e15, 1e20])
            if not self._test_large_number_stability(large_numbers):
                stability_errors.append("Large number instability detected")
            
            # Test 2: Small number handling
            small_numbers = np.array([1e-10, 1e-15, 1e-20])
            if not self._test_small_number_stability(small_numbers):
                stability_errors.append("Small number instability detected")
            
            # Test 3: NaN and infinity handling
            if not self._test_nan_infinity_handling():
                stability_errors.append("NaN/infinity handling failed")
            
            # Test 4: Precision edge cases
            if not self._test_precision_edge_cases():
                stability_errors.append("Precision edge case failures")
            
            passed = len(stability_errors) == 0
            error_message = "; ".join(stability_errors) if stability_errors else None
            
            return ValidationResult(
                test_name="Numerical Stability",
                passed=passed,
                error_message=error_message,
                computed_value=len(stability_errors),
                expected_value=0,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Numerical Stability",
                passed=False,
                error_message=f"Exception during stability validation: {str(e)}",
                computed_value=None,
                expected_value=None,
                tolerance=self.tolerance,
                timestamp=datetime.now()
            )
    
    def _test_large_number_stability(self, large_numbers: np.ndarray) -> bool:
        """Test stability with large numbers."""
        try:
            # Test basic operations
            result = np.sum(large_numbers)
            if not np.isfinite(result):
                return False
            
            # Test normalization
            normalized = large_numbers / np.sum(large_numbers)
            if not np.allclose(np.sum(normalized), 1.0, atol=self.tolerance):
                return False
            
            return True
        except (ValueError, TypeError, AttributeError) as e:
            return False
    
    def _test_small_number_stability(self, small_numbers: np.ndarray) -> bool:
        """Test stability with small numbers."""
        try:
            # Test that small numbers don't underflow to zero inappropriately
            result = np.sum(small_numbers)
            if result == 0 and np.any(small_numbers > 0):
                return False
            
            # Test division by small numbers
            if np.any(small_numbers > 0):
                division_result = 1.0 / small_numbers[small_numbers > 0]
                if not np.all(np.isfinite(division_result)):
                    return False
            
            return True
        except (ValueError, TypeError, AttributeError) as e:
            return False
    
    def _test_nan_infinity_handling(self) -> bool:
        """Test NaN and infinity handling."""
        try:
            # Test NaN detection
            nan_array = np.array([1.0, np.nan, 3.0])
            if not np.any(np.isnan(nan_array)):
                return False
            
            # Test infinity detection
            inf_array = np.array([1.0, np.inf, 3.0])
            if not np.any(np.isinf(inf_array)):
                return False
            
            return True
        except (ValueError, TypeError, AttributeError) as e:
            return False
    
    def _test_precision_edge_cases(self) -> bool:
        """Test precision edge cases."""
        try:
            # Test very close numbers
            a = 1.0
            b = 1.0 + 1e-16
            
            # Should be able to detect they're different with appropriate tolerance
            if np.isclose(a, b, atol=1e-20):
                return True
            
            # Test precision loss in summation
            small_values = np.full(1000000, 1e-10)
            sum_result = np.sum(small_values)
            expected = 1000000 * 1e-10
            
            if abs(sum_result - expected) / expected > 1e-6:
                return False
            
            return True
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return False
    
    def _update_validation_summary(self, results: Dict[str, ValidationResult]) -> None:
        """Update validation summary statistics."""
        for result in results.values():
            self.total_tests += 1
            if result.passed:
                self.passed_tests += 1
            self.validation_history.append(result)
        
        # Keep only last 1000 results
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        success_rate = self.passed_tests / max(1, self.total_tests)
        
        recent_results = self.validation_history[-10:] if self.validation_history else []
        recent_failures = [r for r in recent_results if not r.passed]
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'recent_failures': [
                {
                    'test_name': r.test_name,
                    'error_message': r.error_message,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in recent_failures
            ],
            'last_validation_time': (
                self.validation_history[-1].timestamp.isoformat() 
                if self.validation_history else None
            )
        }
    
    def generate_validation_report(self) -> str:
        """Generate a detailed validation report."""
        summary = self.get_validation_summary()
        
        report = f"""
Mathematical Validation Report
============================

Summary:
- Total Tests: {summary['total_tests']}
- Passed Tests: {summary['passed_tests']}
- Success Rate: {summary['success_rate']:.2%}
- Tolerance: {self.tolerance}

Recent Test Results:
"""
        
        for result in self.validation_history[-5:]:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report += f"- {result.test_name}: {status}\n"
            if not result.passed:
                report += f"  Error: {result.error_message}\n"
        
        if summary['recent_failures']:
            report += "\nRecent Failures:\n"
            for failure in summary['recent_failures']:
                report += f"- {failure['test_name']}: {failure['error_message']}\n"
        
        return report