#!/usr/bin/env python3
"""
Demo script showing how to use the Strategic MARL Latency Benchmark Suite.

This script demonstrates the comprehensive benchmarking capabilities for validating
the <50ms latency target for strategic_marl_component.py.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.test_strategic_marl_latency_benchmark import (
    StrategicMARLBenchmark, 
    LatencyBenchmarkConfig
)


async def demo_single_inference_benchmark():
    """Demonstrate single inference latency benchmarking."""
    print("=" * 60)
    print("Demo: Single Inference Latency Benchmark")
    print("=" * 60)
    
    # Configure benchmark for demo (smaller numbers for faster execution)
    config = LatencyBenchmarkConfig(
        target_latency_ms=50.0,
        p95_latency_ms=75.0,
        p99_latency_ms=100.0,
        warmup_iterations=3,
        benchmark_iterations=20,
        concurrent_users=3
    )
    
    # Create benchmark suite
    benchmark = StrategicMARLBenchmark(config)
    
    print(f"Running single inference benchmark with {config.benchmark_iterations} iterations...")
    print(f"Target latency: {config.target_latency_ms}ms")
    print()
    
    try:
        # Run the benchmark
        result = await benchmark.run_single_inference_benchmark()
        
        # Display results
        print("Results:")
        print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"  P50 Latency: {result.metrics.p50_latency_ms:.2f}ms")
        print(f"  P95 Latency: {result.metrics.p95_latency_ms:.2f}ms")
        print(f"  P99 Latency: {result.metrics.p99_latency_ms:.2f}ms")
        print(f"  Average Latency: {result.metrics.avg_latency_ms:.2f}ms")
        print(f"  Throughput: {result.metrics.throughput_ops_per_sec:.1f} ops/sec")
        print(f"  Error Rate: {result.metrics.error_rate_percent:.1f}%")
        print(f"  Success Count: {result.metrics.success_count}")
        
        if result.failures:
            print("\nFailures:")
            for failure in result.failures:
                print(f"  - {failure}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        return result.passed
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        return False


async def demo_load_testing_benchmark():
    """Demonstrate load testing benchmark."""
    print("\n" + "=" * 60)
    print("Demo: Load Testing Benchmark")
    print("=" * 60)
    
    # Configure for load testing
    config = LatencyBenchmarkConfig(
        target_latency_ms=50.0,
        p95_latency_ms=75.0,
        p99_latency_ms=100.0,
        load_test_duration_sec=15,  # Short duration for demo
        concurrent_users=5,
        min_throughput_ops_per_sec=50.0  # Lower for demo
    )
    
    benchmark = StrategicMARLBenchmark(config)
    
    print(f"Running load test with {config.concurrent_users} concurrent users...")
    print(f"Duration: {config.load_test_duration_sec} seconds")
    print(f"Target throughput: {config.min_throughput_ops_per_sec} ops/sec")
    print()
    
    try:
        # Run load test
        result = await benchmark.run_load_testing_benchmark()
        
        # Display results
        print("Results:")
        print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"  Total Operations: {result.metrics.success_count + result.metrics.failure_count}")
        print(f"  Successful Operations: {result.metrics.success_count}")
        print(f"  Failed Operations: {result.metrics.failure_count}")
        print(f"  Throughput: {result.metrics.throughput_ops_per_sec:.1f} ops/sec")
        print(f"  P95 Latency: {result.metrics.p95_latency_ms:.2f}ms")
        print(f"  Error Rate: {result.metrics.error_rate_percent:.1f}%")
        
        if result.failures:
            print("\nFailures:")
            for failure in result.failures:
                print(f"  - {failure}")
        
        return result.passed
        
    except Exception as e:
        print(f"Load test failed with error: {e}")
        return False


async def demo_performance_regression_detection():
    """Demonstrate performance regression detection."""
    print("\n" + "=" * 60)
    print("Demo: Performance Regression Detection")
    print("=" * 60)
    
    config = LatencyBenchmarkConfig(
        warmup_iterations=2,
        benchmark_iterations=10,
        regression_threshold_percent=20.0
    )
    
    benchmark = StrategicMARLBenchmark(config)
    
    print("Running multiple benchmarks to build baseline...")
    
    try:
        # Run several benchmarks to build history
        for i in range(3):
            print(f"  Running benchmark {i+1}/3...")
            result = await benchmark.run_single_inference_benchmark()
            
            # Check for regression
            regression_detected, messages = benchmark.detect_performance_regression(result)
            
            print(f"  Benchmark {i+1}: P50={result.metrics.p50_latency_ms:.2f}ms")
            if regression_detected:
                print(f"  Regression detected: {messages}")
            else:
                print(f"  No regression: {messages}")
        
        print("\nRegression detection demo completed!")
        return True
        
    except Exception as e:
        print(f"Regression detection demo failed: {e}")
        return False


async def demo_comprehensive_report():
    """Demonstrate comprehensive performance report generation."""
    print("\n" + "=" * 60)
    print("Demo: Comprehensive Performance Report")
    print("=" * 60)
    
    config = LatencyBenchmarkConfig(
        warmup_iterations=2,
        benchmark_iterations=15,
        load_test_duration_sec=10,
        concurrent_users=3
    )
    
    benchmark = StrategicMARLBenchmark(config)
    
    print("Running comprehensive benchmark suite...")
    
    try:
        results = []
        
        # Run single inference benchmark
        print("1. Running single inference benchmark...")
        single_result = await benchmark.run_single_inference_benchmark()
        results.append(single_result)
        
        # Run load testing benchmark
        print("2. Running load testing benchmark...")
        load_result = await benchmark.run_load_testing_benchmark()
        results.append(load_result)
        
        # Generate comprehensive report
        print("3. Generating comprehensive report...")
        report = benchmark.generate_performance_report(results)
        
        # Display report
        print("\n" + "=" * 60)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 60)
        print(report)
        
        # Save report to file
        report_file = "/tmp/strategic_marl_demo_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")
        
        # Overall success
        overall_passed = all(result.passed for result in results)
        print(f"\nOverall Result: {'PASSED' if overall_passed else 'FAILED'}")
        
        return overall_passed
        
    except Exception as e:
        print(f"Comprehensive report demo failed: {e}")
        return False


async def main():
    """Main demo function."""
    print("Strategic MARL Component <50ms Latency Benchmark Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive benchmarking capabilities")
    print("for validating the <50ms latency target for strategic_marl_component.py")
    print()
    
    # Run all demos
    demos = [
        ("Single Inference Benchmark", demo_single_inference_benchmark),
        ("Load Testing Benchmark", demo_load_testing_benchmark),
        ("Performance Regression Detection", demo_performance_regression_detection),
        ("Comprehensive Report", demo_comprehensive_report),
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\nRunning {name}...")
        try:
            success = await demo_func()
            results.append((name, success))
            print(f"{name}: {'SUCCESS' if success else 'FAILED'}")
        except Exception as e:
            print(f"{name}: FAILED - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} demos passed")
    
    # Final message
    if total_passed == len(results):
        print("\n🎉 All demos passed! The benchmark suite is ready for use.")
        print("\nNext steps:")
        print("1. Run 'python -m pytest tests/performance/test_strategic_marl_latency_benchmark.py -v' for full test suite")
        print("2. Use the benchmark suite to validate your Strategic MARL Component performance")
        print("3. Set up continuous benchmarking in your CI/CD pipeline")
    else:
        print("\n⚠️  Some demos failed. Please check the error messages above.")
    
    return total_passed == len(results)


if __name__ == "__main__":
    """Run the demo."""
    success = asyncio.run(main())
    sys.exit(0 if success else 1)