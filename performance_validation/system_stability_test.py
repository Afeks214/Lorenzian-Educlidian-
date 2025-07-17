#!/usr/bin/env python3
"""
System Stability Test for Heavy Load Conditions
===============================================

This script tests system stability under heavy load conditions
including memory pressure, CPU saturation, and concurrent processing.
"""

import os
import sys
import time
import threading
import multiprocessing
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import json
import signal

class SystemStabilityTest:
    """
    Comprehensive system stability testing under heavy load conditions.
    """
    
    def __init__(self):
        self.test_results = {}
        self.stop_flag = threading.Event()
        self.system_metrics = []
        self.monitoring_thread = None
        
        # Test configuration
        self.test_config = {
            'memory_pressure_test_duration': 60,  # seconds
            'cpu_saturation_duration': 30,  # seconds  
            'concurrent_processes': min(4, multiprocessing.cpu_count()),
            'memory_limit_mb': 6000,  # 6GB limit for safety
            'cpu_limit_percent': 95
        }
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signal for graceful shutdown."""
        print("\nReceived interrupt signal. Shutting down gracefully...")
        self.stop_flag.set()
        sys.exit(0)
    
    def _monitor_system_metrics(self):
        """Monitor system metrics during tests."""
        while not self.stop_flag.is_set():
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / (1024**2),
                    'memory_available_mb': psutil.virtual_memory().available / (1024**2),
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                }
                
                self.system_metrics.append(metrics)
                
                # Safety check - stop if memory usage too high
                if metrics['memory_used_mb'] > self.test_config['memory_limit_mb']:
                    print(f"Warning: Memory usage {metrics['memory_used_mb']:.1f}MB exceeds limit")
                    self.stop_flag.set()
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"Error monitoring system metrics: {e}")
                time.sleep(1)
    
    def test_memory_pressure(self) -> Dict:
        """Test system behavior under memory pressure."""
        print("\n=== Testing Memory Pressure ===")
        
        test_start = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)
        
        # Start monitoring
        self.system_metrics = []
        self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        try:
            # Gradually increase memory usage
            memory_blocks = []
            block_size = 50 * 1024 * 1024  # 50MB blocks
            
            duration = self.test_config['memory_pressure_test_duration']
            
            for i in range(duration):
                if self.stop_flag.is_set():
                    break
                
                # Allocate memory block
                try:
                    block = np.random.random(block_size // 8)  # 8 bytes per float64
                    memory_blocks.append(block)
                    
                    current_memory = psutil.virtual_memory().used / (1024**2)
                    memory_increase = current_memory - start_memory
                    
                    print(f"Memory pressure test {i+1}/{duration} - Memory increase: {memory_increase:.1f}MB")
                    
                    # Safety check
                    if memory_increase > 4000:  # 4GB limit
                        print("Memory limit reached, stopping allocation")
                        break
                        
                except MemoryError:
                    print("Memory allocation failed - system limit reached")
                    break
                
                time.sleep(1)
            
            # Hold memory for stability test
            print("Holding memory allocation for stability test...")
            time.sleep(10)
            
            # Gradually release memory
            print("Releasing memory...")
            while memory_blocks:
                del memory_blocks[-1]
                memory_blocks.pop()
                
                if len(memory_blocks) % 10 == 0:
                    current_memory = psutil.virtual_memory().used / (1024**2)
                    memory_decrease = start_memory - current_memory
                    print(f"Memory released: {memory_decrease:.1f}MB")
                
                time.sleep(0.1)
            
            self.stop_flag.set()
            
            test_end = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)
            
            # Analyze metrics
            peak_memory = max([m['memory_used_mb'] for m in self.system_metrics])
            avg_cpu = np.mean([m['cpu_percent'] for m in self.system_metrics])
            
            return {
                'success': True,
                'test_duration': test_end - test_start,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - start_memory,
                'memory_recovery_mb': end_memory - start_memory,
                'avg_cpu_percent': avg_cpu,
                'memory_blocks_allocated': len(memory_blocks),
                'stability_rating': 'stable' if abs(end_memory - start_memory) < 100 else 'memory_leak'
            }
            
        except Exception as e:
            self.stop_flag.set()
            return {
                'success': False,
                'error': str(e),
                'test_duration': time.time() - test_start
            }
    
    def test_cpu_saturation(self) -> Dict:
        """Test system behavior under CPU saturation."""
        print("\n=== Testing CPU Saturation ===")
        
        test_start = time.time()
        
        # Start monitoring
        self.system_metrics = []
        self.stop_flag = threading.Event()
        self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        try:
            # CPU intensive work function
            def cpu_intensive_work(worker_id: int):
                """CPU intensive computation."""
                print(f"Worker {worker_id} starting CPU intensive work...")
                
                local_counter = 0
                start_time = time.time()
                
                while not self.stop_flag.is_set():
                    # Matrix multiplication to stress CPU
                    matrix_a = np.random.random((100, 100))
                    matrix_b = np.random.random((100, 100))
                    result = np.dot(matrix_a, matrix_b)
                    
                    local_counter += 1
                    
                    # Check if we've run long enough
                    if time.time() - start_time > self.test_config['cpu_saturation_duration']:
                        break
                
                print(f"Worker {worker_id} completed {local_counter} operations")
                return local_counter
            
            # Start CPU intensive processes
            processes = []
            num_processes = self.test_config['concurrent_processes']
            
            for i in range(num_processes):
                p = threading.Thread(target=cpu_intensive_work, args=(i,))
                p.daemon = True
                processes.append(p)
                p.start()
            
            # Monitor progress
            duration = self.test_config['cpu_saturation_duration']
            
            for i in range(duration):
                if self.stop_flag.is_set():
                    break
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                print(f"CPU saturation test {i+1}/{duration} - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                
                # Safety check
                if cpu_percent > self.test_config['cpu_limit_percent']:
                    print("CPU usage limit reached")
            
            # Stop all processes
            self.stop_flag.set()
            
            # Wait for processes to complete
            for p in processes:
                p.join(timeout=5)
            
            test_end = time.time()
            
            # Analyze metrics
            if self.system_metrics:
                peak_cpu = max([m['cpu_percent'] for m in self.system_metrics])
                avg_cpu = np.mean([m['cpu_percent'] for m in self.system_metrics])
                avg_memory = np.mean([m['memory_percent'] for m in self.system_metrics])
                
                return {
                    'success': True,
                    'test_duration': test_end - test_start,
                    'peak_cpu_percent': peak_cpu,
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_percent': avg_memory,
                    'concurrent_processes': num_processes,
                    'stability_rating': 'stable' if peak_cpu < 100 else 'cpu_throttled'
                }
            else:
                return {
                    'success': False,
                    'error': 'No metrics collected'
                }
                
        except Exception as e:
            self.stop_flag.set()
            return {
                'success': False,
                'error': str(e),
                'test_duration': time.time() - test_start
            }
    
    def test_concurrent_data_processing(self) -> Dict:
        """Test concurrent data processing stability."""
        print("\n=== Testing Concurrent Data Processing ===")
        
        test_start = time.time()
        
        # Start monitoring
        self.system_metrics = []
        self.stop_flag = threading.Event()
        self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        try:
            def data_processing_worker(worker_id: int):
                """Data processing worker."""
                print(f"Data worker {worker_id} starting...")
                
                records_processed = 0
                
                for batch in range(50):  # Process 50 batches
                    if self.stop_flag.is_set():
                        break
                    
                    # Generate synthetic data
                    data_size = 1000
                    data = {
                        'timestamp': pd.date_range('2023-01-01', periods=data_size, freq='1min'),
                        'price': np.random.normal(16000, 100, data_size),
                        'volume': np.random.randint(1000, 10000, data_size)
                    }
                    
                    df = pd.DataFrame(data)
                    
                    # Process data
                    df['returns'] = df['price'].pct_change()
                    df['volume_ma'] = df['volume'].rolling(window=10).mean()
                    df['volatility'] = df['returns'].rolling(window=20).std()
                    
                    # Simulate more complex processing
                    correlation_matrix = np.corrcoef(df['price'].dropna(), df['volume'].dropna())
                    
                    records_processed += len(df)
                    
                    # Brief pause to simulate real processing
                    time.sleep(0.1)
                
                print(f"Data worker {worker_id} processed {records_processed} records")
                return records_processed
            
            # Start concurrent data processing
            workers = []
            num_workers = 4
            
            for i in range(num_workers):
                worker = threading.Thread(target=data_processing_worker, args=(i,))
                worker.daemon = True
                workers.append(worker)
                worker.start()
            
            # Monitor for 60 seconds
            for i in range(60):
                if self.stop_flag.is_set():
                    break
                
                if self.system_metrics:
                    latest_metrics = self.system_metrics[-1]
                    cpu_percent = latest_metrics['cpu_percent']
                    memory_percent = latest_metrics['memory_percent']
                    
                    print(f"Concurrent processing {i+1}/60 - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                
                time.sleep(1)
            
            # Stop workers
            self.stop_flag.set()
            
            # Wait for workers to complete
            for worker in workers:
                worker.join(timeout=10)
            
            test_end = time.time()
            
            # Analyze metrics
            if self.system_metrics:
                peak_cpu = max([m['cpu_percent'] for m in self.system_metrics])
                peak_memory = max([m['memory_percent'] for m in self.system_metrics])
                avg_cpu = np.mean([m['cpu_percent'] for m in self.system_metrics])
                avg_memory = np.mean([m['memory_percent'] for m in self.system_metrics])
                
                return {
                    'success': True,
                    'test_duration': test_end - test_start,
                    'peak_cpu_percent': peak_cpu,
                    'peak_memory_percent': peak_memory,
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_percent': avg_memory,
                    'concurrent_workers': num_workers,
                    'stability_rating': 'stable' if peak_cpu < 90 and peak_memory < 80 else 'stressed'
                }
            else:
                return {
                    'success': False,
                    'error': 'No metrics collected'
                }
                
        except Exception as e:
            self.stop_flag.set()
            return {
                'success': False,
                'error': str(e),
                'test_duration': time.time() - test_start
            }
    
    def test_error_recovery(self) -> Dict:
        """Test system error recovery capabilities."""
        print("\n=== Testing Error Recovery ===")
        
        test_start = time.time()
        
        recovery_tests = []
        
        # Test 1: Memory allocation failure recovery
        try:
            print("Testing memory allocation failure recovery...")
            
            # Try to allocate large memory block
            try:
                huge_block = np.random.random(1000000000)  # Very large allocation
                del huge_block
                recovery_tests.append({'test': 'memory_allocation', 'result': 'no_failure'})
            except MemoryError:
                # This is expected - test recovery
                recovery_tests.append({'test': 'memory_allocation', 'result': 'recovered'})
            
        except Exception as e:
            recovery_tests.append({'test': 'memory_allocation', 'result': 'failed', 'error': str(e)})
        
        # Test 2: Division by zero handling
        try:
            print("Testing division by zero handling...")
            
            test_data = np.array([1, 2, 3, 0, 5])
            
            # Test division with zero handling
            with np.errstate(divide='ignore', invalid='ignore'):
                result = 1 / test_data
                result = np.where(np.isinf(result), 0, result)
            
            recovery_tests.append({'test': 'division_by_zero', 'result': 'handled'})
            
        except Exception as e:
            recovery_tests.append({'test': 'division_by_zero', 'result': 'failed', 'error': str(e)})
        
        # Test 3: File access error recovery
        try:
            print("Testing file access error recovery...")
            
            # Try to read non-existent file
            try:
                pd.read_csv('nonexistent_file.csv')
                recovery_tests.append({'test': 'file_access', 'result': 'unexpected_success'})
            except FileNotFoundError:
                # Expected error - test recovery
                recovery_tests.append({'test': 'file_access', 'result': 'recovered'})
            
        except Exception as e:
            recovery_tests.append({'test': 'file_access', 'result': 'failed', 'error': str(e)})
        
        test_end = time.time()
        
        # Analyze recovery tests
        successful_recoveries = len([t for t in recovery_tests if t['result'] in ['recovered', 'handled']])
        total_tests = len(recovery_tests)
        
        return {
            'success': True,
            'test_duration': test_end - test_start,
            'recovery_tests': recovery_tests,
            'successful_recoveries': successful_recoveries,
            'total_tests': total_tests,
            'recovery_rate': successful_recoveries / total_tests if total_tests > 0 else 0,
            'stability_rating': 'robust' if successful_recoveries == total_tests else 'fragile'
        }
    
    def run_stability_test_suite(self) -> Dict:
        """Run complete stability test suite."""
        print("System Stability Test Suite")
        print("=" * 40)
        
        suite_start = time.time()
        
        self.test_results = {
            'suite_info': {
                'start_time': datetime.now().isoformat(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': sys.version.split()[0]
                }
            },
            'tests': {}
        }
        
        # Run stability tests
        try:
            self.test_results['tests']['memory_pressure'] = self.test_memory_pressure()
        except Exception as e:
            self.test_results['tests']['memory_pressure'] = {'success': False, 'error': str(e)}
        
        try:
            self.test_results['tests']['cpu_saturation'] = self.test_cpu_saturation()
        except Exception as e:
            self.test_results['tests']['cpu_saturation'] = {'success': False, 'error': str(e)}
        
        try:
            self.test_results['tests']['concurrent_processing'] = self.test_concurrent_data_processing()
        except Exception as e:
            self.test_results['tests']['concurrent_processing'] = {'success': False, 'error': str(e)}
        
        try:
            self.test_results['tests']['error_recovery'] = self.test_error_recovery()
        except Exception as e:
            self.test_results['tests']['error_recovery'] = {'success': False, 'error': str(e)}
        
        suite_end = time.time()
        
        self.test_results['suite_info']['end_time'] = datetime.now().isoformat()
        self.test_results['suite_info']['total_duration'] = suite_end - suite_start
        
        # Generate assessment
        self.test_results['assessment'] = self._generate_stability_assessment()
        
        return self.test_results
    
    def _generate_stability_assessment(self) -> Dict:
        """Generate stability assessment from test results."""
        
        total_tests = 0
        passed_tests = 0
        stability_issues = []
        
        for test_name, result in self.test_results['tests'].items():
            total_tests += 1
            
            if result.get('success', False):
                passed_tests += 1
                
                # Check stability ratings
                stability_rating = result.get('stability_rating', 'unknown')
                if stability_rating not in ['stable', 'robust', 'handled']:
                    stability_issues.append(f"{test_name}: {stability_rating}")
            else:
                stability_issues.append(f"{test_name}: Test failed - {result.get('error', 'Unknown error')}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Overall stability assessment
        if success_rate >= 90 and len(stability_issues) == 0:
            overall_stability = "EXCELLENT"
        elif success_rate >= 80 and len(stability_issues) <= 1:
            overall_stability = "GOOD"
        elif success_rate >= 60:
            overall_stability = "ADEQUATE"
        else:
            overall_stability = "POOR"
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'overall_stability': overall_stability,
            'stability_issues': stability_issues,
            'ready_for_production': overall_stability in ['EXCELLENT', 'GOOD']
        }
    
    def generate_stability_report(self) -> str:
        """Generate stability test report."""
        
        report_lines = []
        
        report_lines.append("# System Stability Test Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # System info
        system_info = self.test_results['suite_info']['system_info']
        report_lines.append("## System Information")
        report_lines.append("")
        for key, value in system_info.items():
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        
        # Assessment
        assessment = self.test_results.get('assessment', {})
        report_lines.append("## Stability Assessment")
        report_lines.append("")
        report_lines.append(f"- Overall Stability: {assessment.get('overall_stability', 'UNKNOWN')}")
        report_lines.append(f"- Production Ready: {assessment.get('ready_for_production', False)}")
        report_lines.append(f"- Test Success Rate: {assessment.get('success_rate', 0):.1f}%")
        report_lines.append(f"- Tests Passed: {assessment.get('passed_tests', 0)}/{assessment.get('total_tests', 0)}")
        report_lines.append("")
        
        # Stability issues
        if assessment.get('stability_issues'):
            report_lines.append("## Stability Issues")
            report_lines.append("")
            for issue in assessment['stability_issues']:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        
        # Test results
        report_lines.append("## Test Results")
        report_lines.append("")
        
        for test_name, result in self.test_results['tests'].items():
            report_lines.append(f"### {test_name}")
            report_lines.append("")
            
            if result.get('success'):
                report_lines.append("- Status: PASSED")
                report_lines.append(f"- Duration: {result.get('test_duration', 0):.2f} seconds")
                
                if 'stability_rating' in result:
                    report_lines.append(f"- Stability Rating: {result['stability_rating']}")
                
                if 'peak_cpu_percent' in result:
                    report_lines.append(f"- Peak CPU: {result['peak_cpu_percent']:.1f}%")
                
                if 'peak_memory_mb' in result:
                    report_lines.append(f"- Peak Memory: {result['peak_memory_mb']:.1f} MB")
                
                if 'recovery_rate' in result:
                    report_lines.append(f"- Recovery Rate: {result['recovery_rate']:.1%}")
                
            else:
                report_lines.append("- Status: FAILED")
                report_lines.append(f"- Error: {result.get('error', 'Unknown error')}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        overall_stability = assessment.get('overall_stability', 'UNKNOWN')
        
        if overall_stability == 'EXCELLENT':
            report_lines.append("- System demonstrates excellent stability under heavy load")
            report_lines.append("- Ready for production deployment")
            report_lines.append("- Consider implementing continuous monitoring")
        elif overall_stability == 'GOOD':
            report_lines.append("- System shows good stability with minor issues")
            report_lines.append("- Address identified stability issues")
            report_lines.append("- Monitor system performance in production")
        else:
            report_lines.append("- System requires stability improvements")
            report_lines.append("- Address critical stability issues before production")
            report_lines.append("- Consider load balancing and resource scaling")
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: str = "performance_validation/results"):
        """Save stability test results."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"stability_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save report
        report_path = os.path.join(output_dir, f"stability_report_{timestamp}.md")
        with open(report_path, 'w') as f:
            f.write(self.generate_stability_report())
        
        print(f"\nStability test results saved:")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")
        
        return json_path, report_path

def main():
    """Main execution function."""
    
    print("System Stability Test for Heavy Load Conditions")
    print("=" * 50)
    
    # Create stability tester
    tester = SystemStabilityTest()
    
    # Run stability tests
    results = tester.run_stability_test_suite()
    
    # Save results
    json_path, report_path = tester.save_results()
    
    # Print summary
    print("\n" + "=" * 50)
    print("STABILITY TEST COMPLETE")
    print("=" * 50)
    
    assessment = results.get('assessment', {})
    
    print(f"Overall Stability: {assessment.get('overall_stability', 'UNKNOWN')}")
    print(f"Production Ready: {assessment.get('ready_for_production', False)}")
    print(f"Success Rate: {assessment.get('success_rate', 0):.1f}%")
    
    if assessment.get('stability_issues'):
        print(f"Stability Issues: {len(assessment['stability_issues'])}")
    
    print(f"\nDetailed report available at: {report_path}")

if __name__ == "__main__":
    main()