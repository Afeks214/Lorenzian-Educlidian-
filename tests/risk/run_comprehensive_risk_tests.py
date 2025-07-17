#!/usr/bin/env python3
"""
Comprehensive Risk Management Integration Test Runner

This script runs all risk management integration tests in a coordinated manner
to demonstrate the complete risk management testing framework.

Test Suites:
1. VaR Model Integration Testing
2. Real-time Risk Monitoring Testing  
3. Correlation Regime Detection Testing
4. Operational Risk Testing
5. Counterparty Risk Testing

Usage:
    python run_comprehensive_risk_tests.py [--suite SUITE_NAME] [--verbose] [--report]
"""

import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test suite configurations
TEST_SUITES = {
    "var_integration": {
        "name": "VaR Model Integration Testing",
        "file": "test_var_integration.py",
        "description": "Tests VaR calculation across multiple asset classes and scenarios",
        "priority": "HIGH",
        "estimated_duration": 120  # seconds
    },
    "realtime_monitoring": {
        "name": "Real-time Risk Monitoring Testing",
        "file": "test_realtime_risk_monitoring.py", 
        "description": "Tests real-time position monitoring and risk limit checking",
        "priority": "HIGH",
        "estimated_duration": 150
    },
    "correlation_regime": {
        "name": "Correlation Regime Detection Testing",
        "file": "test_correlation_regime_detection.py",
        "description": "Tests correlation regime detection and shock identification",
        "priority": "HIGH", 
        "estimated_duration": 180
    },
    "operational_risk": {
        "name": "Operational Risk Testing",
        "file": "test_operational_risk.py",
        "description": "Tests system failure scenarios and recovery procedures",
        "priority": "MEDIUM",
        "estimated_duration": 100
    },
    "counterparty_risk": {
        "name": "Counterparty Risk Testing", 
        "file": "test_counterparty_risk.py",
        "description": "Tests credit exposure and counterparty default scenarios",
        "priority": "MEDIUM",
        "estimated_duration": 90
    }
}

class TestResult:
    """Test result container"""
    
    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.start_time = None
        self.end_time = None
        self.duration = 0
        self.passed = False
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
        self.error_message = None
        self.output = ""
        
    def start(self):
        """Mark test start"""
        self.start_time = datetime.now()
        
    def finish(self, passed: bool, output: str = "", error: str = ""):
        """Mark test completion"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.passed = passed
        self.output = output
        self.error_message = error
        
    def parse_pytest_output(self, output: str):
        """Parse pytest output to extract test statistics"""
        lines = output.split('\n')
        
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Parse line like "5 passed, 2 failed in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed,':
                        self.tests_passed = int(parts[i-1])
                    elif part == 'failed':
                        self.tests_failed = int(parts[i-1])
                    elif part == 'skipped':
                        self.tests_skipped = int(parts[i-1])
                        
        self.tests_run = self.tests_passed + self.tests_failed + self.tests_skipped


class RiskTestRunner:
    """Comprehensive risk test runner"""
    
    def __init__(self, verbose: bool = False, generate_report: bool = False):
        self.verbose = verbose
        self.generate_report = generate_report
        self.results: Dict[str, TestResult] = {}
        self.start_time = None
        self.end_time = None
        
    def run_test_suite(self, suite_name: str) -> TestResult:
        """Run a single test suite"""
        
        suite_config = TEST_SUITES[suite_name]
        result = TestResult(suite_name)
        
        print(f"\n{'='*60}")
        print(f"🧪 Running: {suite_config['name']}")
        print(f"📄 Description: {suite_config['description']}")
        print(f"⏱️  Estimated Duration: {suite_config['estimated_duration']}s")
        print(f"{'='*60}")
        
        result.start()
        
        try:
            # Construct pytest command
            test_file = Path(__file__).parent / suite_config['file']
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--asyncio-mode=auto",
                "--disable-warnings"
            ]
            
            if self.verbose:
                cmd.append("-s")
            
            # Run test
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config['estimated_duration'] * 2  # 2x timeout buffer
            )
            
            # Parse results
            success = process.returncode == 0
            output = process.stdout
            error = process.stderr
            
            result.finish(success, output, error)
            result.parse_pytest_output(output)
            
            # Print summary
            if success:
                print(f"✅ {suite_config['name']} PASSED")
                print(f"   Tests: {result.tests_passed} passed, {result.tests_failed} failed")
                print(f"   Duration: {result.duration:.1f}s")
            else:
                print(f"❌ {suite_config['name']} FAILED")
                print(f"   Tests: {result.tests_passed} passed, {result.tests_failed} failed")
                print(f"   Duration: {result.duration:.1f}s")
                if self.verbose and error:
                    print(f"   Error: {error}")
                    
        except subprocess.TimeoutExpired:
            result.finish(False, "", "Test suite timed out")
            print(f"⏰ {suite_config['name']} TIMED OUT")
            
        except Exception as e:
            result.finish(False, "", str(e))
            print(f"💥 {suite_config['name']} ERROR: {str(e)}")
            
        return result
    
    def run_all_suites(self, suites: Optional[List[str]] = None):
        """Run all test suites"""
        
        if suites is None:
            suites = list(TEST_SUITES.keys())
        
        self.start_time = datetime.now()
        
        print("🎯 AGENT 5 MISSION: Risk Management Integration Testing")
        print("=" * 70)
        print("Testing comprehensive risk management across all system components")
        print(f"Suites to run: {', '.join(suites)}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run each suite
        for suite_name in suites:
            if suite_name not in TEST_SUITES:
                print(f"❌ Unknown test suite: {suite_name}")
                continue
                
            self.results[suite_name] = self.run_test_suite(suite_name)
            
            # Small delay between suites
            time.sleep(1)
        
        self.end_time = datetime.now()
        
        # Print final summary
        self.print_summary()
        
        # Generate report if requested
        if self.generate_report:
            self.generate_test_report()
    
    def print_summary(self):
        """Print test execution summary"""
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("📊 RISK MANAGEMENT INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r.passed)
        failed_suites = total_suites - passed_suites
        
        total_tests = sum(r.tests_run for r in self.results.values())
        total_passed = sum(r.tests_passed for r in self.results.values())
        total_failed = sum(r.tests_failed for r in self.results.values())
        total_skipped = sum(r.tests_skipped for r in self.results.values())
        
        print(f"📈 Overall Results:")
        print(f"   Test Suites: {passed_suites}/{total_suites} passed ({passed_suites/total_suites:.1%})")
        print(f"   Individual Tests: {total_passed}/{total_tests} passed ({total_passed/total_tests:.1%})")
        print(f"   Total Duration: {total_duration:.1f}s")
        print()
        
        # Suite-by-suite results
        print("📋 Suite Results:")
        for suite_name, result in self.results.items():
            suite_config = TEST_SUITES[suite_name]
            status = "✅ PASS" if result.passed else "❌ FAIL"
            
            print(f"   {status} {suite_config['name']}")
            print(f"      Tests: {result.tests_passed}✅ {result.tests_failed}❌ {result.tests_skipped}⏭️")
            print(f"      Duration: {result.duration:.1f}s")
            
            if not result.passed and result.error_message:
                print(f"      Error: {result.error_message}")
            print()
        
        # Risk coverage assessment
        print("🛡️  Risk Coverage Assessment:")
        
        coverage_areas = {
            "var_integration": "Market Risk (VaR)",
            "realtime_monitoring": "Real-time Risk Controls", 
            "correlation_regime": "Correlation & Regime Risk",
            "operational_risk": "Operational Risk",
            "counterparty_risk": "Counterparty & Credit Risk"
        }
        
        for suite_name, area in coverage_areas.items():
            if suite_name in self.results:
                status = "✅ COVERED" if self.results[suite_name].passed else "❌ NOT COVERED"
                print(f"   {status} {area}")
        
        print()
        
        # Overall mission assessment
        success_rate = passed_suites / total_suites
        
        if success_rate >= 0.8:
            print("🎉 MISSION STATUS: SUCCESS")
            print("   Risk management integration testing completed successfully!")
            print("   System is ready for production deployment.")
        elif success_rate >= 0.6:
            print("⚠️  MISSION STATUS: PARTIAL SUCCESS")
            print("   Most risk management components tested successfully.")
            print("   Address failing tests before production deployment.")
        else:
            print("❌ MISSION STATUS: FAILURE")
            print("   Critical risk management components failed testing.")
            print("   System requires significant fixes before deployment.")
        
        print("=" * 70)
    
    def generate_test_report(self):
        """Generate detailed test report"""
        
        report_data = {
            "mission": "Risk Management Integration Testing",
            "agent": "AGENT 5",
            "timestamp": datetime.now().isoformat(),
            "execution_time": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "summary": {
                "total_suites": len(self.results),
                "passed_suites": sum(1 for r in self.results.values() if r.passed),
                "failed_suites": sum(1 for r in self.results.values() if not r.passed),
                "total_tests": sum(r.tests_run for r in self.results.values()),
                "passed_tests": sum(r.tests_passed for r in self.results.values()),
                "failed_tests": sum(r.tests_failed for r in self.results.values()),
                "skipped_tests": sum(r.tests_skipped for r in self.results.values())
            },
            "suites": {}
        }
        
        # Add detailed suite results
        for suite_name, result in self.results.items():
            suite_config = TEST_SUITES[suite_name]
            
            report_data["suites"][suite_name] = {
                "name": suite_config["name"],
                "description": suite_config["description"],
                "priority": suite_config["priority"],
                "estimated_duration": suite_config["estimated_duration"],
                "actual_duration": result.duration,
                "passed": result.passed,
                "tests_run": result.tests_run,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "tests_skipped": result.tests_skipped,
                "error_message": result.error_message
            }
        
        # Save report
        report_file = Path(__file__).parent / f"risk_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📊 Test report generated: {report_file}")
    
    def get_success_rate(self) -> float:
        """Get overall test success rate"""
        if not self.results:
            return 0.0
        
        passed = sum(1 for r in self.results.values() if r.passed)
        return passed / len(self.results)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Risk Management Integration Test Runner"
    )
    
    parser.add_argument(
        "--suite", 
        choices=list(TEST_SUITES.keys()),
        help="Run specific test suite only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true", 
        help="Generate detailed test report"
    )
    
    parser.add_argument(
        "--list-suites", "-l",
        action="store_true",
        help="List available test suites"
    )
    
    args = parser.parse_args()
    
    # List suites if requested
    if args.list_suites:
        print("Available Risk Management Test Suites:")
        print("=" * 50)
        
        for suite_name, config in TEST_SUITES.items():
            print(f"📋 {suite_name}")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Priority: {config['priority']}")
            print(f"   Estimated Duration: {config['estimated_duration']}s")
            print()
        
        return
    
    # Create test runner
    runner = RiskTestRunner(
        verbose=args.verbose,
        generate_report=args.report
    )
    
    # Run tests
    if args.suite:
        runner.run_all_suites([args.suite])
    else:
        runner.run_all_suites()
    
    # Exit with appropriate code
    success_rate = runner.get_success_rate()
    
    if success_rate >= 0.8:
        sys.exit(0)  # Success
    elif success_rate >= 0.6:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # Failure


if __name__ == "__main__":
    main()