#!/usr/bin/env python3
"""
Test runner for the Trading System Controller testing framework.

This script provides a comprehensive test runner for the master switch system,
with options for different test categories and performance benchmarking.
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
import tempfile


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode == 0:
        print(f"✅ SUCCESS ({duration:.2f}s)")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ FAILED ({duration:.2f}s)")
        print(f"Exit code: {result.returncode}")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    
    return result


def run_unit_tests(verbose=False):
    """Run unit tests for core functionality."""
    cmd = "python -m pytest tests/system_control/test_master_switch.py"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "Unit Tests - Core Functionality")
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = "python -m pytest tests/system_control/test_integration.py"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "Integration Tests")
    return result.returncode == 0


def run_safety_tests(verbose=False):
    """Run safety mechanism tests."""
    cmd = "python -m pytest tests/system_control/test_safety_checks.py"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "Safety Mechanism Tests")
    return result.returncode == 0


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = "python -m pytest tests/system_control/test_performance.py"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "Performance Tests")
    return result.returncode == 0


def run_stress_tests(verbose=False):
    """Run stress tests."""
    cmd = "python -m pytest tests/system_control/ -m stress --stress"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "Stress Tests")
    return result.returncode == 0


def run_all_tests(verbose=False):
    """Run all tests."""
    cmd = "python -m pytest tests/system_control/"
    if verbose:
        cmd += " -v"
    
    result = run_command(cmd, "All Tests")
    return result.returncode == 0


def run_coverage_report():
    """Generate coverage report."""
    cmd = "python -m pytest tests/system_control/ --cov=src.core.trading_system_controller --cov-report=html --cov-report=term"
    result = run_command(cmd, "Coverage Report")
    return result.returncode == 0


def run_benchmark_suite():
    """Run performance benchmark suite."""
    cmd = "python -m pytest tests/system_control/test_performance.py -v --benchmark-only"
    result = run_command(cmd, "Performance Benchmark Suite")
    return result.returncode == 0


def validate_test_environment():
    """Validate test environment setup."""
    print("\n🔍 Validating test environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required modules
    required_modules = [
        'pytest', 'psutil', 'concurrent.futures', 'threading', 'time',
        'tempfile', 'json', 'statistics', 'unittest.mock'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
        return False
    
    # Check test files exist
    test_files = [
        "tests/system_control/test_master_switch.py",
        "tests/system_control/test_integration.py",
        "tests/system_control/test_safety_checks.py",
        "tests/system_control/test_performance.py",
        "src/core/trading_system_controller.py"
    ]
    
    missing_files = []
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ Test environment validation successful!")
    return True


def generate_test_report():
    """Generate comprehensive test report."""
    print("\n📊 Generating comprehensive test report...")
    
    # Create temporary directory for reports
    with tempfile.TemporaryDirectory() as temp_dir:
        report_file = os.path.join(temp_dir, "test_report.json")
        
        # Run tests with JSON report
        cmd = f"python -m pytest tests/system_control/ --json-report --json-report-file={report_file}"
        result = run_command(cmd, "Test Report Generation")
        
        if result.returncode == 0 and os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Generate summary
            summary = report_data.get('summary', {})
            print(f"\n📋 Test Summary:")
            print(f"  Total tests: {summary.get('total', 0)}")
            print(f"  Passed: {summary.get('passed', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")
            print(f"  Skipped: {summary.get('skipped', 0)}")
            print(f"  Duration: {summary.get('duration', 0):.2f}s")
            
            # Save report to current directory
            final_report = "system_control_test_report.json"
            with open(final_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n💾 Full report saved to: {final_report}")
            
            return True
        else:
            print("❌ Failed to generate test report")
            return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Trading System Controller Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit              # Run unit tests only
  python run_tests.py --integration       # Run integration tests only
  python run_tests.py --safety           # Run safety tests only
  python run_tests.py --performance      # Run performance tests only
  python run_tests.py --stress           # Run stress tests only
  python run_tests.py --all              # Run all tests
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --benchmark        # Run performance benchmarks
  python run_tests.py --validate         # Validate test environment
  python run_tests.py --report           # Generate comprehensive report
        """
    )
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--safety", action="store_true", help="Run safety tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--stress", action="store_true", help="Run stress tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    
    args = parser.parse_args()
    
    # If no specific test type is chosen, run validation
    if not any([args.unit, args.integration, args.safety, args.performance, 
                args.stress, args.all, args.coverage, args.benchmark, args.validate, args.report]):
        args.validate = True
    
    success = True
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"🚀 Trading System Controller Test Runner")
    print(f"Working directory: {os.getcwd()}")
    print(f"Test directory: {project_root}/tests/system_control/")
    
    # Validate environment first
    if args.validate or any([args.unit, args.integration, args.safety, args.performance, args.all]):
        if not validate_test_environment():
            print("\n❌ Environment validation failed. Please fix issues before running tests.")
            sys.exit(1)
    
    # Run tests based on arguments
    if args.unit:
        success &= run_unit_tests(args.verbose)
    
    if args.integration:
        success &= run_integration_tests(args.verbose)
    
    if args.safety:
        success &= run_safety_tests(args.verbose)
    
    if args.performance:
        success &= run_performance_tests(args.verbose)
    
    if args.stress:
        success &= run_stress_tests(args.verbose)
    
    if args.all:
        success &= run_all_tests(args.verbose)
    
    if args.coverage:
        success &= run_coverage_report()
    
    if args.benchmark:
        success &= run_benchmark_suite()
    
    if args.report:
        success &= generate_test_report()
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All requested tests completed successfully!")
        print("\n🎉 System Control Testing Framework: PASSED")
    else:
        print("❌ Some tests failed or encountered errors.")
        print("\n⚠️  System Control Testing Framework: FAILED")
        sys.exit(1)
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()