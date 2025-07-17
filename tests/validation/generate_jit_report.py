"""
Generate JIT Validation Report

Consolidates JIT compatibility and performance test results into a comprehensive report.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JITValidationReporter:
    """Generate comprehensive JIT validation report."""
    
    def __init__(self):
        """Initialize reporter."""
        self.compatibility_results = None
        self.performance_results = None
        self.report_status = "JIT_VALIDATION_PASSED"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("📄 Generating JIT validation report")
        
        try:
            # Run compatibility tests
            logger.info("Running JIT compatibility tests...")
            self.compatibility_results = self._run_compatibility_tests()
            
            # Run performance benchmarks
            logger.info("Running JIT performance benchmarks...")
            self.performance_results = self._run_performance_benchmarks()
            
            # Generate consolidated report
            report = self._create_consolidated_report()
            
            # Print report
            self._print_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate JIT validation report: {e}")
            self.report_status = "JIT_VALIDATION_ERROR"
            return {
                "status": self.report_status,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _run_compatibility_tests(self) -> Dict[str, Any]:
        """Run JIT compatibility tests."""
        try:
            # Import and run compatibility tests
            from test_jit_compatibility import JITCompatibilityTester
            
            tester = JITCompatibilityTester()
            results = tester.test_all_models()
            
            if not results["all_passed"]:
                self.report_status = "JIT_COMPATIBILITY_FAILED"
            
            return results
            
        except Exception as e:
            logger.error(f"Compatibility test failed: {e}")
            self.report_status = "JIT_COMPATIBILITY_FAILED"
            return {
                "all_passed": False,
                "error": str(e)
            }
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run JIT performance benchmarks."""
        try:
            # Import and run performance benchmarks
            from benchmark_jit_performance import JITPerformanceBenchmark
            
            benchmark = JITPerformanceBenchmark()
            results = benchmark.benchmark_all_models()
            
            if not results["all_targets_met"]:
                if self.report_status == "JIT_VALIDATION_PASSED":
                    self.report_status = "JIT_PERFORMANCE_DEGRADED"
            
            return results
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            if self.report_status == "JIT_VALIDATION_PASSED":
                self.report_status = "JIT_PERFORMANCE_DEGRADED"
            return {
                "all_targets_met": False,
                "error": str(e)
            }
    
    def _create_consolidated_report(self) -> Dict[str, Any]:
        """Create consolidated validation report."""
        report = {
            "timestamp": self._get_timestamp(),
            "status": self.report_status,
            "summary": {
                "overall_status": self.report_status,
                "compatibility_passed": self.compatibility_results.get("all_passed", False) if self.compatibility_results else False,
                "performance_passed": self.performance_results.get("all_targets_met", False) if self.performance_results else False,
                "deployment_ready": self.report_status == "JIT_VALIDATION_PASSED"
            },
            "compatibility_results": self.compatibility_results,
            "performance_results": self.performance_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Compatibility recommendations
        if self.compatibility_results and not self.compatibility_results.get("all_passed", False):
            recommendations.append({
                "type": "CRITICAL",
                "message": "JIT compilation failed for some models. Review model architecture for TorchScript compatibility.",
                "action": "Fix model compatibility issues before deployment"
            })
        
        # Performance recommendations
        if self.performance_results and not self.performance_results.get("all_targets_met", False):
            avg_speedup = self.performance_results.get("average_speedup_factor", 0)
            if avg_speedup < 1.2:
                recommendations.append({
                    "type": "WARNING",
                    "message": f"JIT speedup ({avg_speedup:.2f}x) is below target (1.2x). Consider model optimization.",
                    "action": "Optimize model architecture or review JIT compilation settings"
                })
        
        # General recommendations
        if self.report_status == "JIT_VALIDATION_PASSED":
            recommendations.append({
                "type": "SUCCESS",
                "message": "All JIT validation tests passed. Models are ready for deployment.",
                "action": "Proceed with deployment"
            })
        
        return recommendations
    
    def _print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        print("\n" + "=" * 80)
        print("JIT VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['status']}")
        print(f"Deployment Ready: {report['summary']['deployment_ready']}")
        
        print("\n" + "-" * 40)
        print("COMPATIBILITY RESULTS")
        print("-" * 40)
        
        if self.compatibility_results:
            comp = self.compatibility_results
            print(f"Status: {'PASSED' if comp.get('all_passed', False) else 'FAILED'}")
            print(f"Models tested: {comp.get('total_models', 0)}")
            print(f"Successful: {comp.get('successful_models', 0)}")
            print(f"Failed: {comp.get('failed_models', 0)}")
            
            if comp.get('failed_models', 0) > 0:
                print("Failed models:")
                for model in comp.get('failed_model_names', []):
                    print(f"  - {model}")
        else:
            print("No compatibility results available")
        
        print("\n" + "-" * 40)
        print("PERFORMANCE RESULTS")
        print("-" * 40)
        
        if self.performance_results:
            perf = self.performance_results
            print(f"Status: {'PASSED' if perf.get('all_targets_met', False) else 'FAILED'}")
            print(f"Average inference time: {perf.get('average_inference_time_ms', 0):.2f}ms")
            print(f"Average speedup: {perf.get('average_speedup_factor', 0):.2f}x")
            
            targets = perf.get('performance_targets', {})
            if targets:
                print("Performance targets:")
                for target, value in targets.items():
                    print(f"  {target}: {value}")
        else:
            print("No performance results available")
        
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        
        for rec in report.get('recommendations', []):
            print(f"[{rec['type']}] {rec['message']}")
            print(f"Action: {rec['action']}")
            print()
        
        print("=" * 80)
        print(f"FINAL STATUS: {report['status']}")
        print("=" * 80)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.utcnow().isoformat() + "Z"

def main():
    """Main function for standalone execution."""
    reporter = JITValidationReporter()
    report = reporter.generate_report()
    
    # Exit with appropriate code based on status
    status = report.get("status", "JIT_VALIDATION_ERROR")
    
    if status == "JIT_VALIDATION_PASSED":
        sys.exit(0)
    elif status == "JIT_PERFORMANCE_DEGRADED":
        sys.exit(1)  # Warning level
    else:
        sys.exit(2)  # Critical failure

if __name__ == "__main__":
    main()