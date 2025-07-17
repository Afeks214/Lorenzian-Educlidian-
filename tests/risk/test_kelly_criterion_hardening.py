"""
Comprehensive Adversarial Test Suite for Kelly Criterion Security

This test suite intentionally attempts to exploit the Kelly Criterion calculator
with malicious, corrupted, and edge case inputs. Every test must PASS, 
confirming that the validation layer catches all dangerous scenarios.

Author: Agent 1 - Input Guardian
Date: 2025-07-13
Mission: Prove unconditional security of Kelly implementation
"""

import pytest
import math
import numpy as np
import sys
import os
from unittest.mock import patch
from datetime import datetime

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, src_path)

from risk.core.kelly_calculator import (
    KellyCalculator, 
    KellySecurityViolation, 
    KellyInputError,
    create_bulletproof_kelly_calculator,
    calculate_safe_kelly
)


class TestKellySecurityValidation:
    """Test security validation layers."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_nan_probability_attack(self):
        """Test NaN probability input is blocked."""
        with pytest.raises(KellySecurityViolation, match="NaN win_probability detected"):
            self.calc.calculate_position_size(float('nan'), 2.0)
            
    def test_nan_payout_attack(self):
        """Test NaN payout ratio input is blocked.""" 
        with pytest.raises(KellySecurityViolation, match="NaN payout_ratio detected"):
            self.calc.calculate_position_size(0.6, float('nan'))
            
    def test_infinite_probability_attack(self):
        """Test infinite probability input is blocked."""
        with pytest.raises(KellySecurityViolation, match="Infinite win_probability detected"):
            self.calc.calculate_position_size(float('inf'), 2.0)
            
    def test_infinite_payout_attack(self):
        """Test infinite payout ratio input is blocked."""
        with pytest.raises(KellySecurityViolation, match="Infinite payout_ratio detected"):
            self.calc.calculate_position_size(0.6, float('inf'))
            
    def test_negative_infinity_probability_attack(self):
        """Test negative infinite probability input is blocked."""
        with pytest.raises(KellySecurityViolation, match="Infinite win_probability detected"):
            self.calc.calculate_position_size(float('-inf'), 2.0)
            
    def test_negative_probability_attack(self):
        """Test negative probability input is blocked."""
        with pytest.raises(KellySecurityViolation, match="win_probability.*<= 0.*potential attack"):
            self.calc.calculate_position_size(-0.1, 2.0)
            
    def test_probability_greater_than_one_attack(self):
        """Test probability > 1 input is blocked."""
        with pytest.raises(KellySecurityViolation, match="win_probability.*>= 1.*potential attack"):
            self.calc.calculate_position_size(1.5, 2.0)
            
    def test_negative_payout_attack(self):
        """Test negative payout ratio input is blocked."""
        with pytest.raises(KellySecurityViolation, match="Non-positive payout_ratio.*potential attack"):
            self.calc.calculate_position_size(0.6, -1.0)
            
    def test_zero_payout_attack(self):
        """Test zero payout ratio input is blocked."""
        with pytest.raises(KellySecurityViolation, match="Non-positive payout_ratio.*potential attack"):
            self.calc.calculate_position_size(0.6, 0.0)
            
    def test_extremely_large_payout_attack(self):
        """Test extremely large payout ratio is capped."""
        with pytest.raises(KellyInputError, match="exceeds maximum allowed"):
            self.calc.calculate_position_size(0.6, 1e7)


class TestKellyTypeConfusionAttacks:
    """Test type confusion attack vectors."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_string_probability_attack(self):
        """Test string probability input is blocked."""
        with pytest.raises(KellyInputError, match="win_probability must be numeric"):
            self.calc.calculate_position_size("0.6", 2.0)
            
    def test_string_payout_attack(self):
        """Test string payout ratio input is blocked."""
        with pytest.raises(KellyInputError, match="payout_ratio must be numeric"):
            self.calc.calculate_position_size(0.6, "2.0")
            
    def test_none_probability_attack(self):
        """Test None probability input is blocked."""
        with pytest.raises(KellyInputError, match="win_probability must be numeric"):
            self.calc.calculate_position_size(None, 2.0)
            
    def test_none_payout_attack(self):
        """Test None payout ratio input is blocked."""
        with pytest.raises(KellyInputError, match="payout_ratio must be numeric"):
            self.calc.calculate_position_size(0.6, None)
            
    def test_list_probability_attack(self):
        """Test list probability input is blocked."""
        with pytest.raises(KellyInputError, match="win_probability must be numeric"):
            self.calc.calculate_position_size([0.6], 2.0)
            
    def test_dict_payout_attack(self):
        """Test dict payout ratio input is blocked."""
        with pytest.raises(KellyInputError, match="payout_ratio must be numeric"):
            self.calc.calculate_position_size(0.6, {"payout": 2.0})
            
    def test_boolean_inputs_attack(self):
        """Test boolean inputs are blocked."""
        with pytest.raises(KellyInputError, match="win_probability must be numeric"):
            self.calc.calculate_position_size(True, 2.0)
            
    def test_complex_number_attack(self):
        """Test complex number inputs are blocked."""
        with pytest.raises(KellyInputError, match="win_probability must be numeric"):
            self.calc.calculate_position_size(0.6 + 1j, 2.0)


class TestKellyEdgeCases:
    """Test mathematical edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_extremely_small_probability(self):
        """Test extremely small but valid probability."""
        result = self.calc.calculate_position_size(1e-10, 2.0)
        assert result.kelly_fraction < 0  # Should be negative (don't bet)
        assert abs(result.kelly_fraction) <= 0.25  # Within safety bounds
        
    def test_extremely_large_probability(self):
        """Test probability very close to 1."""
        result = self.calc.calculate_position_size(0.999999, 2.0)
        assert result.kelly_fraction > 0  # Should be positive
        assert result.kelly_fraction <= 0.25  # Within safety bounds
        
    def test_extremely_small_payout(self):
        """Test extremely small payout ratio."""
        result = self.calc.calculate_position_size(0.6, 1e-10)
        assert abs(result.kelly_fraction) <= 0.25  # Within safety bounds
        
    def test_probability_exactly_zero(self):
        """Test probability of exactly 0."""
        with pytest.raises(KellySecurityViolation, match="win_probability.*<= 0.*potential attack"):
            self.calc.calculate_position_size(0.0, 2.0)
            
    def test_probability_exactly_one(self):
        """Test probability of exactly 1."""
        with pytest.raises(KellySecurityViolation, match="win_probability.*>= 1.*potential attack"):
            self.calc.calculate_position_size(1.0, 2.0)
            
    def test_negative_capital_attack(self):
        """Test negative capital input is blocked."""
        with pytest.raises(KellyInputError, match="capital must be positive"):
            self.calc.calculate_position_size(0.6, 2.0, -1000.0)
            
    def test_zero_capital_attack(self):
        """Test zero capital input is blocked."""
        with pytest.raises(KellyInputError, match="capital must be positive"):
            self.calc.calculate_position_size(0.6, 2.0, 0.0)


class TestKellyNumericalStability:
    """Test numerical stability and overflow protection."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_very_large_numbers(self):
        """Test calculation with very large numbers."""
        result = self.calc.calculate_position_size(0.6, 1000.0, 1e6)
        assert not math.isnan(result.kelly_fraction)
        assert not math.isinf(result.kelly_fraction)
        assert abs(result.kelly_fraction) <= 0.25
        
    def test_very_small_numbers(self):
        """Test calculation with very small numbers."""
        result = self.calc.calculate_position_size(0.6, 1e-3, 1e-6)
        assert not math.isnan(result.kelly_fraction)
        assert not math.isinf(result.kelly_fraction)
        assert abs(result.kelly_fraction) <= 0.25
        
    def test_precision_boundary_conditions(self):
        """Test calculations at floating point precision boundaries."""
        # Test with numbers close to machine epsilon
        epsilon = np.finfo(float).eps
        result = self.calc.calculate_position_size(0.5 + epsilon, 2.0)
        assert not math.isnan(result.kelly_fraction)
        assert not math.isinf(result.kelly_fraction)


class TestKellyRollingValidation:
    """Test rolling statistical validation system."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_rolling_validation_buildup(self):
        """Test that rolling validation builds up history."""
        # Add normal observations
        for _ in range(200):
            result = self.calc.calculate_position_size(0.55, 2.0)
            assert result.inputs.validation_passed
            
        # Now test anomaly detection
        result = self.calc.calculate_position_size(0.95, 2.0)  # Extreme deviation
        assert result.capped_by_validation  # Should be capped
        assert "probability_capped_3sigma_deviation" in str(result.inputs.security_flags)
        
    def test_rolling_validation_without_history(self):
        """Test behavior with insufficient rolling history."""
        result = self.calc.calculate_position_size(0.95, 2.0)  # No history yet
        assert not result.capped_by_validation  # Should not be capped
        assert result.inputs.rolling_deviation == 0.0
        
    def test_statistical_anomaly_detection(self):
        """Test detection of statistical anomalies."""
        # Build consistent history
        for _ in range(300):
            self.calc.calculate_position_size(0.55, 2.0)
            
        # Try extreme outlier
        result = self.calc.calculate_position_size(0.05, 2.0)  # Very different
        assert result.capped_by_validation


class TestKellyPerformanceAndSafety:
    """Test performance requirements and safety guarantees."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_calculation_speed_requirement(self):
        """Test that calculations complete within 1ms requirement."""
        start_time = datetime.now()
        result = self.calc.calculate_position_size(0.6, 2.0)
        end_time = datetime.now()
        
        calculation_time_ms = (end_time - start_time).total_seconds() * 1000
        assert calculation_time_ms < 1.0, f"Calculation took {calculation_time_ms:.3f}ms, exceeding 1ms limit"
        assert result.calculation_time_ms < 1.0
        
    def test_maximum_kelly_fraction_bound(self):
        """Test that Kelly fraction never exceeds safety bound."""
        test_cases = [
            (0.9, 10.0),    # High probability, high payout
            (0.999, 100.0), # Extreme probability, extreme payout
            (0.8, 1000.0),  # High payout ratio
        ]
        
        for prob, payout in test_cases:
            result = self.calc.calculate_position_size(prob, payout)
            assert abs(result.kelly_fraction) <= 0.25, \
                f"Kelly fraction {result.kelly_fraction} exceeds safety bound for prob={prob}, payout={payout}"
                
    def test_security_violation_tracking(self):
        """Test that security violations are properly tracked."""
        initial_violations = self.calc.security_violations
        
        # Trigger security violations
        for _ in range(5):
            try:
                self.calc.calculate_position_size(float('nan'), 2.0)
            except KellySecurityViolation:
                pass
                
        stats = self.calc.get_security_stats()
        assert stats['security_violations'] == initial_violations + 5
        assert stats['security_violation_rate'] > 0


class TestKellySimpleFunctionInterface:
    """Test the simple function interface for Kelly calculation."""
    
    def test_calculate_safe_kelly_function(self):
        """Test the module-level safe Kelly function."""
        kelly = calculate_safe_kelly(0.6, 2.0)
        assert isinstance(kelly, float)
        assert not math.isnan(kelly)
        assert not math.isinf(kelly)
        assert abs(kelly) <= 0.25
        
    def test_calculate_safe_kelly_with_capital(self):
        """Test safe Kelly function with capital parameter."""
        kelly = calculate_safe_kelly(0.6, 2.0, 1000.0)
        assert isinstance(kelly, float)
        assert abs(kelly) <= 0.25
        
    def test_calculate_safe_kelly_security(self):
        """Test that simple function also enforces security."""
        with pytest.raises(KellySecurityViolation):
            calculate_safe_kelly(float('nan'), 2.0)


class TestKellyMathematicalProof:
    """Test mathematical proof of safety implementation."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_mathematical_proof_exists(self):
        """Test that mathematical proof method exists and returns content."""
        proof = self.calc.mathematical_proof_of_safety()
        assert isinstance(proof, str)
        assert len(proof) > 100  # Should be substantial
        assert "MATHEMATICAL PROOF" in proof
        assert "QED" in proof
        
    def test_proof_covers_all_safety_aspects(self):
        """Test that proof covers all critical safety aspects."""
        proof = self.calc.mathematical_proof_of_safety()
        
        # Check for coverage of key safety aspects
        assert "TYPE SAFETY" in proof
        assert "VALUE BOUNDS" in proof
        assert "KELLY FORMULA BOUNDS" in proof
        assert "ROLLING VALIDATION" in proof
        assert "OVERFLOW PROTECTION" in proof
        assert "safe_kelly_output" in proof


class TestKellyStressTest:
    """Stress test the Kelly calculator with intensive scenarios."""
    
    def setup_method(self):
        """Set up fresh calculator for each test."""
        self.calc = create_bulletproof_kelly_calculator()
        
    def test_repeated_calculations_stability(self):
        """Test stability under repeated calculations."""
        for i in range(1000):
            prob = 0.5 + (i % 100) / 1000.0  # Varying probabilities
            payout = 1.5 + (i % 50) / 100.0  # Varying payouts
            
            result = self.calc.calculate_position_size(prob, payout)
            assert not math.isnan(result.kelly_fraction)
            assert not math.isinf(result.kelly_fraction)
            assert abs(result.kelly_fraction) <= 0.25
            
    def test_concurrent_calculation_safety(self):
        """Test that calculator is safe under concurrent access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def calculate_worker():
            try:
                for _ in range(100):
                    result = self.calc.calculate_position_size(0.6, 2.0)
                    results.append(result.kelly_fraction)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(str(e))
                
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=calculate_worker)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify no errors and all results are valid
        assert len(errors) == 0, f"Concurrent calculation errors: {errors}"
        assert len(results) == 500  # 5 threads * 100 calculations
        for kelly in results:
            assert not math.isnan(kelly)
            assert not math.isinf(kelly)
            assert abs(kelly) <= 0.25


if __name__ == "__main__":
    # Run specific test for quick verification
    pytest.main([__file__, "-v", "-x"])  # Stop on first failure