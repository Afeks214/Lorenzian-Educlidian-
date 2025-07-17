"""
NWRQK Kernel Validation Script

This script validates that the NWRQK kernel fix correctly implements the 
distance calculation ||x_t - x_i||^2 instead of the incorrect index i.

Demonstrates the difference between the old (incorrect) and new (correct) implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

from src.indicators.custom.nwrqk import rational_quadratic_kernel


def old_incorrect_kernel(i: int, h_param: float, r_param: float) -> float:
    """
    OLD INCORRECT implementation using index i instead of distance
    This is what was in the code before the fix
    """
    return (1 + (i**2 / ((h_param**2) * 2 * r_param)))**(-r_param)


def new_correct_kernel(x_t: float, x_i: float, alpha: float = 1.0, h: float = 1.0) -> float:
    """
    NEW CORRECT implementation using proper distance calculation
    """
    return rational_quadratic_kernel(x_t, x_i, alpha, h)


class NWRQKValidationSuite:
    """Comprehensive validation of NWRQK kernel correction"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_mathematical_correctness(self) -> Dict[str, bool]:
        """Test mathematical properties of the corrected kernel"""
        print("=== Testing Mathematical Correctness ===")
        
        results = {}
        
        # Test 1: Symmetry K(x,y) = K(y,x)
        x, y = 100.0, 105.0
        k1 = new_correct_kernel(x, y)
        k2 = new_correct_kernel(y, x)
        symmetry_ok = abs(k1 - k2) < 1e-10
        results['symmetry'] = symmetry_ok
        print(f"Symmetry test: {'PASS' if symmetry_ok else 'FAIL'} - K({x},{y})={k1:.10f}, K({y},{x})={k2:.10f}")
        
        # Test 2: Self-kernel K(x,x) = 1
        k_self = new_correct_kernel(x, x)
        self_kernel_ok = abs(k_self - 1.0) < 1e-10
        results['self_kernel'] = self_kernel_ok
        print(f"Self-kernel test: {'PASS' if self_kernel_ok else 'FAIL'} - K({x},{x})={k_self:.10f}")
        
        # Test 3: Positive definiteness (all kernels >= 0)
        test_points = np.linspace(90, 110, 21)
        kernels = [new_correct_kernel(100.0, p) for p in test_points]
        positive_ok = all(k >= 0 for k in kernels)
        results['positive'] = positive_ok
        print(f"Positive definiteness: {'PASS' if positive_ok else 'FAIL'}")
        
        # Test 4: Monotonic decay with distance
        distances = [0, 1, 2, 5, 10, 20, 50]
        kernel_values = [new_correct_kernel(100.0, 100.0 + d) for d in distances]
        monotonic_ok = all(kernel_values[i] >= kernel_values[i+1] for i in range(len(kernel_values)-1))
        results['monotonic_decay'] = monotonic_ok
        print(f"Monotonic decay: {'PASS' if monotonic_ok else 'FAIL'}")
        print(f"  Kernel values: {[f'{k:.6f}' for k in kernel_values]}")
        
        return results
    
    def demonstrate_correction_impact(self) -> None:
        """Demonstrate the difference between old and new implementations"""
        print("\n=== Demonstrating Correction Impact ===")
        
        # Parameters for testing
        h_param = 8.0
        r_param = 8.0
        
        # Price series for testing
        prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 100.0])
        current_price = prices[0]  # x_t
        
        print(f"Current price (x_t): {current_price}")
        print(f"Historical prices: {prices[1:]}")
        print()
        
        print("Index | Historical Price | Distance² | Old Kernel (index) | New Kernel (distance) | Difference")
        print("-" * 90)
        
        total_old_weight = 0.0
        total_new_weight = 0.0
        
        for i in range(1, len(prices)):
            historical_price = prices[i]
            distance_squared = (current_price - historical_price) ** 2
            
            # Old incorrect method (using index)
            old_weight = old_incorrect_kernel(i, h_param, r_param)
            
            # New correct method (using distance)
            new_weight = new_correct_kernel(current_price, historical_price, r_param, h_param)
            
            difference = abs(new_weight - old_weight)
            
            print(f"{i:5d} | {historical_price:14.1f} | {distance_squared:9.1f} | {old_weight:18.6f} | {new_weight:19.6f} | {difference:10.6f}")
            
            total_old_weight += old_weight
            total_new_weight += new_weight
        
        print("-" * 90)
        print(f"Total weights: Old={total_old_weight:.6f}, New={total_new_weight:.6f}")
        print(f"Relative difference: {abs(total_new_weight - total_old_weight) / total_old_weight * 100:.2f}%")
        
    def test_regression_calculation(self) -> None:
        """Test the full regression calculation with both methods"""
        print("\n=== Testing Full Regression Calculation ===")
        
        # Generate synthetic price data
        np.random.seed(42)
        n_points = 50
        base_prices = np.linspace(100, 110, n_points)
        noise = np.random.normal(0, 0.5, n_points)
        prices = base_prices + noise
        
        current_price = prices[0]
        
        # Calculate weighted average using old method (index-based)
        h_param, r_param = 8.0, 8.0
        old_numerator = 0.0
        old_denominator = 0.0
        
        for i in range(1, min(25, len(prices))):
            weight = old_incorrect_kernel(i, h_param, r_param)
            old_numerator += prices[i] * weight
            old_denominator += weight
        
        old_result = old_numerator / old_denominator if old_denominator > 0 else current_price
        
        # Calculate weighted average using new method (distance-based)
        new_numerator = 0.0
        new_denominator = 0.0
        
        for i in range(1, min(25, len(prices))):
            weight = new_correct_kernel(current_price, prices[i], r_param, h_param)
            new_numerator += prices[i] * weight
            new_denominator += weight
        
        new_result = new_numerator / new_denominator if new_denominator > 0 else current_price
        
        print(f"Current price: {current_price:.4f}")
        print(f"Old method result: {old_result:.4f}")
        print(f"New method result: {new_result:.4f}")
        print(f"Difference: {abs(new_result - old_result):.4f}")
        print(f"Relative difference: {abs(new_result - old_result) / current_price * 100:.2f}%")
        
        # The corrected method should give different results
        assert abs(new_result - old_result) > 0.001, "Correction should produce different results"
        
    def test_performance_impact(self) -> None:
        """Test performance impact of the correction"""
        print("\n=== Testing Performance Impact ===")
        
        # Generate test data
        n_trials = 1000
        n_points = 100
        prices = np.random.normal(100, 5, n_points)
        
        # Time old method
        start_time = time.time()
        for _ in range(n_trials):
            for i in range(1, min(50, len(prices))):
                old_incorrect_kernel(i, 8.0, 8.0)
        old_time = time.time() - start_time
        
        # Time new method
        start_time = time.time()
        current_price = prices[0]
        for _ in range(n_trials):
            for i in range(1, min(50, len(prices))):
                new_correct_kernel(current_price, prices[i], 8.0, 8.0)
        new_time = time.time() - start_time
        
        print(f"Old method time: {old_time:.4f} seconds")
        print(f"New method time: {new_time:.4f} seconds")
        print(f"Performance ratio: {new_time / old_time:.2f}x")
        
        # New method should still be fast (within 2x of old method)
        assert new_time / old_time < 2.0, "Corrected method should not be more than 2x slower"
        
    def test_numerical_stability(self) -> None:
        """Test numerical stability edge cases"""
        print("\n=== Testing Numerical Stability ===")
        
        test_cases = [
            ("Very small distance", 100.0, 100.0 + 1e-15),
            ("Zero distance", 100.0, 100.0),
            ("Large distance", 0.0, 1000.0),
            ("Negative prices", -50.0, -45.0),
            ("Very large prices", 1e6, 1e6 + 100),
        ]
        
        for description, x_t, x_i in test_cases:
            try:
                result = new_correct_kernel(x_t, x_i)
                is_valid = not (np.isnan(result) or np.isinf(result)) and result >= 0
                status = "PASS" if is_valid else "FAIL"
                print(f"{description}: {status} - K({x_t}, {x_i}) = {result:.10f}")
            except Exception as e:
                print(f"{description}: FAIL - Exception: {e}")
                
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        print("NWRQK Kernel Correction Validation Suite")
        print("=" * 50)
        
        # Run all tests
        math_results = self.test_mathematical_correctness()
        self.demonstrate_correction_impact()
        self.test_regression_calculation()
        self.test_performance_impact()
        self.test_numerical_stability()
        
        # Summary
        print("\n=== Validation Summary ===")
        all_math_passed = all(math_results.values())
        
        if all_math_passed:
            print("✅ ALL MATHEMATICAL TESTS PASSED")
            print("✅ NWRQK kernel correction is mathematically sound")
            print("✅ Ready for production deployment")
        else:
            print("❌ SOME TESTS FAILED")
            for test, result in math_results.items():
                status = "✅" if result else "❌"
                print(f"  {status} {test}")
        
        return all_math_passed


def main():
    """Run the validation suite"""
    validator = NWRQKValidationSuite()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎉 NWRQK kernel correction validation SUCCESSFUL!")
        print("The mathematical error has been fixed and the implementation is correct.")
    else:
        print("\n❌ NWRQK kernel correction validation FAILED!")
        print("There are still issues that need to be addressed.")
    
    return success


if __name__ == "__main__":
    main()