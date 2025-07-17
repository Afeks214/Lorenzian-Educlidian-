#!/usr/bin/env python3
"""
Matrix Assembly Test Suite Runner

This script runs all matrix assembly tests and provides comprehensive
reporting on test results, performance metrics, and coverage.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_file: str) -> Dict[str, Any]:
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    start_time = time.time()
    
    # Run pytest with detailed output
    cmd = [
        sys.executable, '-m', 'pytest',
        str(Path(__file__).parent / test_file),
        '-v',
        '--tb=short',
        '--durations=10',
        '--capture=no'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        return {
            'test_file': test_file,
            'duration': end_time - start_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'test_file': test_file,
            'duration': 300,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Test suite timed out after 5 minutes',
            'success': False
        }
    except Exception as e:
        return {
            'test_file': test_file,
            'duration': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def main():
    """Run all matrix assembly tests."""
    print("AGENT 3 MISSION: Matrix Assemblers & Normalizers Testing")
    print("=" * 60)
    print("Starting comprehensive matrix assembly test suite...")
    
    # Test files to run
    test_files = [
        'test_normalizers.py',
        'test_assembler_30m.py', 
        'test_assembler_5m.py',
        'test_matrix_integration.py',
        'test_performance_validation.py'
    ]
    
    # Run all test suites
    results = []
    total_start_time = time.time()
    
    for test_file in test_files:
        result = run_test_suite(test_file)
        results.append(result)
        
        if result['success']:
            print(f"✅ {test_file} - PASSED ({result['duration']:.2f}s)")
        else:
            print(f"❌ {test_file} - FAILED ({result['duration']:.2f}s)")
            if result['stderr']:
                print(f"Error: {result['stderr']}")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Generate summary report
    print("\n" + "="*60)
    print("MATRIX ASSEMBLY TEST SUITE SUMMARY")
    print("="*60)
    
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = len(results) - passed_tests
    
    print(f"Total Test Suites: {len(results)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    # Detailed results
    for result in results:
        status = "PASSED" if result['success'] else "FAILED"
        print(f"\n{result['test_file']}: {status} ({result['duration']:.2f}s)")
        
        if not result['success']:
            print(f"  Error: {result['stderr']}")
            
        # Extract key metrics from stdout
        if result['stdout']:
            lines = result['stdout'].split('\n')
            for line in lines:
                if 'passed' in line or 'failed' in line or 'error' in line:
                    if '::' not in line:  # Skip individual test cases
                        print(f"  {line}")
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for result in results:
        if result['success']:
            print(f"{result['test_file']}: {result['duration']:.2f}s")
    
    # Generate JSON report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration': total_duration,
        'summary': {
            'total_suites': len(results),
            'passed': passed_tests,
            'failed': failed_tests
        },
        'results': results
    }
    
    report_file = Path(__file__).parent / 'test_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Mission completion assessment
    print("\n" + "="*60)
    print("AGENT 3 MISSION ASSESSMENT")
    print("="*60)
    
    if failed_tests == 0:
        print("🎯 MISSION COMPLETE - ALL TESTS PASSED")
        print("\n✅ Matrix Assembly System Validation:")
        print("   - 30-minute strategic matrix assembler: VALIDATED")
        print("   - 5-minute tactical matrix assembler: VALIDATED")
        print("   - Normalization algorithms: VALIDATED")
        print("   - Pipeline integration: VALIDATED")
        print("   - Performance requirements: VALIDATED")
        print("   - Memory efficiency: VALIDATED")
        
        print("\n📊 Key Achievements:")
        print("   - Comprehensive feature extraction testing")
        print("   - Window processing and circular buffer validation")
        print("   - Statistical normalization accuracy verification")
        print("   - Real-time performance benchmarks")
        print("   - Memory leak detection and resource management")
        print("   - End-to-end pipeline integration testing")
        
        return 0
    else:
        print("❌ MISSION INCOMPLETE - SOME TESTS FAILED")
        print(f"\n{failed_tests} test suite(s) failed:")
        for result in results:
            if not result['success']:
                print(f"   - {result['test_file']}")
        
        print("\n🔧 Action Required:")
        print("   Review failed tests and fix implementation issues")
        print("   Re-run tests after fixes are applied")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())