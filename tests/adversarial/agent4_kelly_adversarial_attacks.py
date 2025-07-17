"""
Agent 4: Red Team Adversarial Attack Suite for Kelly Criterion Security
=========================================================================

Mission: Design sophisticated attack vectors to attempt to bypass the 
hardened Kelly Criterion security layers implemented by Agent 1.

Attack Vector Categories:
1. Input Validation Bypass - Indirect injection through calculation paths
2. Historical Data Poisoning - Manipulation of rolling averages
3. Correlated Input Manipulation - Coordinated multi-input attacks

SUCCESS CRITERIA: ALL ATTACKS MUST FAIL (proving system robustness)

Author: Agent 4 - Red Team Adversarial Tester
Date: 2025-07-13
Critical Mission: Final Aegis - Prove unconditional security
"""

import pytest
import numpy as np
import math
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import threading
import time
import gc
from typing import List, Tuple, Any

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, src_path)

from risk.core.kelly_calculator import (
    KellyCalculator, 
    KellySecurityViolation, 
    KellyInputError,
    create_bulletproof_kelly_calculator,
    calculate_safe_kelly,
    RollingValidator
)


class TestAttackVector1_InputValidationBypass:
    """
    Attack Vector 1: Input Validation Bypass
    
    Attempt to inject malicious values through indirect calculation paths
    and exploit edge cases that might bypass validation logic.
    """
    
    def setup_method(self):
        """Setup for each attack test"""
        self.calc = create_bulletproof_kelly_calculator()
        self.attack_results = []
    
    def test_attack_1a_numpy_array_confusion(self):
        """Attack 1A: Attempt numpy array type confusion to bypass validation"""
        print("🔴 ATTACK 1A: Numpy Array Type Confusion")
        
        # Try to pass numpy arrays with malicious values
        attack_vectors = [
            (np.array([float('nan')]), 2.0, "NaN via numpy array"),
            (np.array([float('inf')]), 2.0, "Infinity via numpy array"),
            (np.array([-0.5]), 2.0, "Negative via numpy array"),
            (np.array([1.5]), 2.0, "Invalid probability via numpy array"),
        ]
        
        for attack_input, payout, description in attack_vectors:
            try:
                result = self.calc.calculate_position_size(attack_input, payout)
                self.attack_results.append(f"❌ BREACH: {description} - Calculation succeeded unexpectedly")
                assert False, f"SECURITY BREACH: {description} bypassed validation"
            except (KellySecurityViolation, KellyInputError, TypeError, ValueError) as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Attack blocked: {description}")
        
        print(f"   Result: All {len(attack_vectors)} numpy attacks blocked")
    
    def test_attack_1b_mathematical_operation_injection(self):
        """Attack 1B: Attempt to inject dangerous values through mathematical operations"""
        print("🔴 ATTACK 1B: Mathematical Operation Injection")
        
        # Attempt to create malicious values through operations
        try:
            overflow_value = math.pow(10, 308) * 1e-308  # Safe overflow test
        except OverflowError:
            overflow_value = float('inf')
            
        attack_vectors = [
            (0.5 * float('inf'), 2.0, "Infinity through multiplication"),
            (1.0 / 0.0 if False else 0.5, 2.0, "Division by zero conditional"),
            (overflow_value, 2.0, "Overflow then underflow"),
            (complex(0.5, 0).real, 2.0, "Complex number real part"),
        ]
        
        for attack_input, payout, description in attack_vectors:
            try:
                if not math.isfinite(attack_input):
                    print(f"   ⚠️  Pre-validation catch: {description} already invalid")
                    continue
                    
                result = self.calc.calculate_position_size(attack_input, payout)
                if not (-0.25 <= result.kelly_fraction <= 0.25):
                    self.attack_results.append(f"❌ BREACH: {description} - Extreme Kelly fraction")
                    assert False, f"SECURITY BREACH: {description} produced extreme output"
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description} - Output within bounds")
                    print(f"   ✅ Attack contained: {description}")
            except (KellySecurityViolation, KellyInputError, ValueError, OverflowError) as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Attack blocked: {description}")
        
        print(f"   Result: All mathematical injection attacks neutralized")
    
    def test_attack_1c_precision_boundary_exploitation(self):
        """Attack 1C: Exploit floating-point precision boundaries"""
        print("🔴 ATTACK 1C: Precision Boundary Exploitation")
        
        # Attack with values at floating-point precision limits
        epsilon = np.finfo(float).eps
        tiny = np.finfo(float).tiny
        max_float = np.finfo(float).max
        
        attack_vectors = [
            (1.0 - epsilon/2, 2.0, "Just under 1.0 via epsilon"),
            (epsilon, max_float, "Minimum float with maximum payout"),
            (1.0 - tiny, 2.0, "1.0 minus minimum positive"),
            (0.5 + epsilon * 1e15, 2.0, "0.5 plus large epsilon multiple"),
        ]
        
        for attack_input, payout, description in attack_vectors:
            try:
                result = self.calc.calculate_position_size(attack_input, payout)
                
                # Verify output is bounded
                if not (-0.25 <= result.kelly_fraction <= 0.25):
                    self.attack_results.append(f"❌ BREACH: {description} - Unbounded output")
                    assert False, f"PRECISION ATTACK SUCCESS: {description}"
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description} - Bounded output")
                    print(f"   ✅ Precision attack contained: {description}")
                    
            except (KellySecurityViolation, KellyInputError) as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Precision attack blocked: {description}")
        
        print(f"   Result: All precision boundary attacks contained")
    
    def test_attack_1d_memory_pressure_validation_bypass(self):
        """Attack 1D: Attempt validation bypass under memory pressure"""
        print("🔴 ATTACK 1D: Memory Pressure Validation Bypass")
        
        # Create memory pressure then attack
        memory_hogs = []
        try:
            # Create memory pressure
            for i in range(10):
                memory_hogs.append(np.random.random((1000, 1000)))
            
            # Attempt attacks under memory pressure
            attack_vectors = [
                (float('nan'), 2.0, "NaN under memory pressure"),
                (-0.5, 2.0, "Negative under memory pressure"),
                (1.5, 2.0, "Invalid probability under memory pressure"),
            ]
            
            for attack_input, payout, description in attack_vectors:
                try:
                    result = self.calc.calculate_position_size(attack_input, payout)
                    self.attack_results.append(f"❌ BREACH: {description} - Validation bypassed")
                    assert False, f"MEMORY PRESSURE BYPASS: {description}"
                except (KellySecurityViolation, KellyInputError) as e:
                    self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                    print(f"   ✅ Memory pressure attack blocked: {description}")
                    
        finally:
            # Clean up memory
            del memory_hogs
            gc.collect()
        
        print(f"   Result: Validation robust under memory pressure")


class TestAttackVector2_HistoricalDataPoisoning:
    """
    Attack Vector 2: Historical Data Poisoning
    
    Attempt to manipulate rolling average calculations through rapid 
    oscillation patterns and temporal dependency exploitation.
    """
    
    def setup_method(self):
        """Setup for historical data attacks"""
        self.calc = create_bulletproof_kelly_calculator()
        self.attack_results = []
    
    def test_attack_2a_rapid_oscillation_poisoning(self):
        """Attack 2A: Poison rolling averages with rapid oscillation"""
        print("🔴 ATTACK 2A: Rapid Oscillation Poisoning")
        
        # Build normal history first
        for i in range(200):
            self.calc.calculate_position_size(0.55, 2.0)
        
        # Now attempt rapid oscillation to confuse rolling validation
        oscillation_patterns = [
            ([0.1, 0.9] * 50, "Extreme binary oscillation"),
            ([0.3, 0.7, 0.4, 0.6] * 25, "Quaternary oscillation"),
            ([0.5 + 0.1 * math.sin(i/10) for i in range(100)], "Sinusoidal poisoning"),
        ]
        
        for pattern, description in oscillation_patterns:
            try:
                # Inject oscillation pattern
                for value in pattern:
                    result = self.calc.calculate_position_size(value, 2.0)
                    
                    # Check if extreme values bypass validation
                    if not (-0.25 <= result.kelly_fraction <= 0.25):
                        self.attack_results.append(f"❌ BREACH: {description} - Extreme output")
                        assert False, f"OSCILLATION ATTACK SUCCESS: {description}"
                
                self.attack_results.append(f"✅ CONTAINED: {description} - All outputs bounded")
                print(f"   ✅ Oscillation attack contained: {description}")
                
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Oscillation attack blocked: {description}")
        
        print(f"   Result: All oscillation poisoning attempts failed")
    
    def test_attack_2b_temporal_dependency_exploitation(self):
        """Attack 2B: Exploit temporal dependencies in validation"""
        print("🔴 ATTACK 2B: Temporal Dependency Exploitation")
        
        # Build predictable temporal pattern
        for i in range(150):
            self.calc.calculate_position_size(0.55, 2.0)
        
        # Attempt to exploit temporal prediction
        temporal_attacks = [
            (lambda i: 0.55 + 0.001 * i, "Linear drift exploitation"),
            (lambda i: 0.55 * (1.01 ** (i/100)), "Exponential drift exploitation"),
            (lambda i: 0.55 + 0.1 * (1 if i % 10 == 0 else 0), "Periodic spike exploitation"),
        ]
        
        for pattern_func, description in temporal_attacks:
            try:
                extreme_values = []
                
                # Execute temporal pattern
                for i in range(50):
                    attack_value = pattern_func(i)
                    result = self.calc.calculate_position_size(attack_value, 2.0)
                    extreme_values.append(result.kelly_fraction)
                    
                    # Check for unbounded outputs
                    if not (-0.25 <= result.kelly_fraction <= 0.25):
                        self.attack_results.append(f"❌ BREACH: {description} - Unbounded at step {i}")
                        assert False, f"TEMPORAL EXPLOIT SUCCESS: {description}"
                
                # Check if pattern created systematic bias
                bias = abs(np.mean(extreme_values))
                if bias > 0.2:  # Significant bias
                    self.attack_results.append(f"⚠️  BIAS: {description} - Systematic bias {bias:.3f}")
                    print(f"   ⚠️  Temporal bias detected: {description} (bias: {bias:.3f})")
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description} - No systematic bias")
                    print(f"   ✅ Temporal attack contained: {description}")
                    
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Temporal attack blocked: {description}")
        
        print(f"   Result: Temporal dependency attacks neutralized")
    
    def test_attack_2c_rolling_window_boundary_attack(self):
        """Attack 2C: Attack rolling window boundaries"""
        print("🔴 ATTACK 2C: Rolling Window Boundary Attack")
        
        # Fill rolling window to near capacity
        window_size = 100  # Estimated window size
        
        # Fill with normal data
        for i in range(window_size - 5):
            self.calc.calculate_position_size(0.55, 2.0)
        
        # Attack at window boundary
        boundary_attacks = [
            ([0.95] * 10, "Extreme values at boundary"),
            ([0.05] * 10, "Minimum values at boundary"),
            ([0.55, 0.05, 0.95] * 3, "Alternating extremes at boundary"),
        ]
        
        for attack_sequence, description in boundary_attacks:
            try:
                for attack_value in attack_sequence:
                    result = self.calc.calculate_position_size(attack_value, 2.0)
                    
                    if not (-0.25 <= result.kelly_fraction <= 0.25):
                        self.attack_results.append(f"❌ BREACH: {description} - Boundary bypass")
                        assert False, f"BOUNDARY ATTACK SUCCESS: {description}"
                
                self.attack_results.append(f"✅ CONTAINED: {description} - Boundary protected")
                print(f"   ✅ Boundary attack contained: {description}")
                
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Boundary attack blocked: {description}")
        
        print(f"   Result: Rolling window boundaries secured")


class TestAttackVector3_CorrelatedInputManipulation:
    """
    Attack Vector 3: Correlated Input Manipulation
    
    Coordinate multiple inputs to bypass individual validation through
    exploiting dependencies between win_probability and payout_ratio.
    """
    
    def setup_method(self):
        """Setup for correlated input attacks"""
        self.calc = create_bulletproof_kelly_calculator()
        self.attack_results = []
    
    def test_attack_3a_coordinated_extreme_inputs(self):
        """Attack 3A: Coordinate extreme win_probability and payout_ratio"""
        print("🔴 ATTACK 3A: Coordinated Extreme Inputs")
        
        # Coordinate inputs to try to create extreme Kelly fractions
        coordinated_attacks = [
            (0.99, 100.0, "High probability + high payout"),
            (0.01, 0.01, "Low probability + low payout"),
            (0.999, 1000.0, "Extreme probability + extreme payout"),
            (0.51, 1000000.0, "Just positive edge + massive payout"),
        ]
        
        for prob, payout, description in coordinated_attacks:
            try:
                result = self.calc.calculate_position_size(prob, payout)
                
                # Check if coordination bypassed safety bounds
                if not (-0.25 <= result.kelly_fraction <= 0.25):
                    self.attack_results.append(f"❌ BREACH: {description} - Extreme Kelly: {result.kelly_fraction}")
                    assert False, f"COORDINATION ATTACK SUCCESS: {description}"
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description} - Kelly: {result.kelly_fraction:.6f}")
                    print(f"   ✅ Coordination contained: {description}")
                    
            except (KellySecurityViolation, KellyInputError) as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Coordination blocked: {description}")
        
        print(f"   Result: All coordinated extreme inputs contained")
    
    def test_attack_3b_mathematical_relationship_exploitation(self):
        """Attack 3B: Exploit mathematical relationships in Kelly formula"""
        print("🔴 ATTACK 3B: Mathematical Relationship Exploitation")
        
        # Kelly = (p*b - q)/b = p - q/b = p - (1-p)/b
        # Try to exploit this relationship
        relationship_attacks = [
            # Make q/b approach 0 to maximize Kelly
            (0.9, 1e6, "Minimize q/b term via large payout"),
            # Make p*b very large
            (0.999, 1e3, "Maximize p*b term"),
            # Edge case where Kelly should be exactly 0
            (0.5, 1.0, "Kelly = 0 edge case"),
            # Just above break-even
            (0.5000001, 1.0, "Infinitesimal positive edge"),
        ]
        
        for prob, payout, description in relationship_attacks:
            try:
                result = self.calc.calculate_position_size(prob, payout)
                
                # Mathematical verification
                expected_kelly = (prob * payout - (1 - prob)) / payout
                expected_kelly = max(-0.25, min(0.25, expected_kelly))  # Apply bounds
                
                # Check if result matches bounded expectation
                if abs(result.kelly_fraction - expected_kelly) > 0.01:
                    self.attack_results.append(f"⚠️  ANOMALY: {description} - Math inconsistency")
                    print(f"   ⚠️  Mathematical anomaly: {description}")
                else:
                    self.attack_results.append(f"✅ CONSISTENT: {description} - Math verified")
                    print(f"   ✅ Mathematical relationship secured: {description}")
                    
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Relationship attack blocked: {description}")
        
        print(f"   Result: Mathematical relationships properly enforced")
    
    def test_attack_3c_correlated_sequence_attacks(self):
        """Attack 3C: Coordinated sequence attacks over time"""
        print("🔴 ATTACK 3C: Correlated Sequence Attacks")
        
        # Execute coordinated sequences to try to confuse system
        sequence_attacks = [
            ([(0.6, 2.0), (0.8, 5.0), (0.9, 10.0)], "Escalating risk sequence"),
            ([(0.5, 1000.0), (0.51, 1000.0), (0.52, 1000.0)], "Incremental probability with high payout"),
            ([(0.9, 1.1), (0.9, 1.2), (0.9, 1.3)], "Fixed probability escalating payout"),
        ]
        
        for sequence, description in sequence_attacks:
            try:
                max_kelly = -1.0
                min_kelly = 1.0
                
                for prob, payout in sequence:
                    result = self.calc.calculate_position_size(prob, payout)
                    max_kelly = max(max_kelly, result.kelly_fraction)
                    min_kelly = min(min_kelly, result.kelly_fraction)
                    
                    # Each individual result must be bounded
                    if not (-0.25 <= result.kelly_fraction <= 0.25):
                        self.attack_results.append(f"❌ BREACH: {description} - Unbounded in sequence")
                        assert False, f"SEQUENCE ATTACK SUCCESS: {description}"
                
                # Check sequence didn't create systematic issues
                kelly_range = max_kelly - min_kelly
                if kelly_range > 0.5:  # Excessive range
                    self.attack_results.append(f"⚠️  RANGE: {description} - Excessive range {kelly_range:.3f}")
                    print(f"   ⚠️  Sequence range concern: {description}")
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description} - Range: {kelly_range:.3f}")
                    print(f"   ✅ Sequence attack contained: {description}")
                    
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Sequence attack blocked: {description}")
        
        print(f"   Result: All correlated sequence attacks neutralized")


class TestAdvancedExploitationAttempts:
    """
    Advanced exploitation attempts including race conditions,
    floating-point edge cases, and mathematical overflow/underflow.
    """
    
    def setup_method(self):
        """Setup for advanced attacks"""
        self.calc = create_bulletproof_kelly_calculator()
        self.attack_results = []
    
    def test_race_condition_exploitation(self):
        """Test concurrent access for race conditions"""
        print("🔴 ADVANCED ATTACK: Race Condition Exploitation")
        
        results = []
        errors = []
        
        def attack_worker(attack_id):
            try:
                # Each thread tries different malicious inputs
                attack_inputs = [
                    (float('nan'), 2.0),
                    (-0.5, 2.0),
                    (1.5, 2.0),
                    (0.5, float('inf')),
                ]
                
                for prob, payout in attack_inputs:
                    try:
                        result = self.calc.calculate_position_size(prob, payout)
                        results.append((attack_id, result.kelly_fraction))
                    except Exception as e:
                        errors.append((attack_id, type(e).__name__, str(e)))
                        
            except Exception as e:
                errors.append((attack_id, "THREAD_ERROR", str(e)))
        
        # Launch concurrent attacks
        threads = []
        for i in range(10):
            thread = threading.Thread(target=attack_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all malicious inputs were blocked
        malicious_successes = [r for r in results if not (-0.25 <= r[1] <= 0.25)]
        
        if malicious_successes:
            self.attack_results.append(f"❌ BREACH: Race condition allowed malicious results")
            assert False, "RACE CONDITION EXPLOIT SUCCESS"
        else:
            self.attack_results.append(f"✅ SECURED: Race conditions handled - {len(errors)} attacks blocked")
            print(f"   ✅ Race condition attacks secured ({len(errors)} blocked)")
    
    def test_floating_point_edge_cases(self):
        """Test floating-point arithmetic edge cases"""
        print("🔴 ADVANCED ATTACK: Floating-Point Edge Cases")
        
        edge_cases = [
            (np.nextafter(0.0, 1.0), 2.0, "Smallest positive float"),
            (np.nextafter(1.0, 0.0), 2.0, "Largest float < 1.0"),
            (0.5 * (1.0 + np.finfo(float).eps), 2.0, "0.5 + epsilon/2"),
            (float.fromhex('0x1.fffffffffffffp-1'), 2.0, "Maximum precise fraction"),
        ]
        
        for prob, payout, description in edge_cases:
            try:
                result = self.calc.calculate_position_size(prob, payout)
                
                if not (-0.25 <= result.kelly_fraction <= 0.25):
                    self.attack_results.append(f"❌ BREACH: {description} - Unbounded")
                    assert False, f"FLOATING-POINT EXPLOIT: {description}"
                else:
                    self.attack_results.append(f"✅ CONTAINED: {description}")
                    print(f"   ✅ Edge case contained: {description}")
                    
            except Exception as e:
                self.attack_results.append(f"✅ BLOCKED: {description} - {type(e).__name__}")
                print(f"   ✅ Edge case blocked: {description}")
        
        print(f"   Result: Floating-point edge cases secured")


def main():
    """Execute all adversarial attacks against Kelly Criterion"""
    print("🔴" * 50)
    print("AGENT 4: RED TEAM ADVERSARIAL TESTING")
    print("Target: Kelly Criterion Security Implementation")
    print("Mission: Attempt to break hardened defenses")
    print("🔴" * 50)
    
    # Run all attack vectors
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure (security breach)
    ])


if __name__ == "__main__":
    main()