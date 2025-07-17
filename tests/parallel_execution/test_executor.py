"""
Advanced Parallel Test Execution System
Agent 2 Mission: Parallel Execution & Test Distribution

This module implements sophisticated parallel test execution with intelligent
distribution strategies, resource management, and performance optimization.
"""

import os
import psutil
import time
import json
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestExecutionMetrics:
    """Metrics for test execution performance"""
    test_name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    worker_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerResource:
    """Resource allocation for test workers"""
    worker_id: str
    cpu_affinity: List[int]
    memory_limit: int  # MB
    gpu_id: Optional[int] = None
    status: str = "idle"  # idle, running, failed, terminated


class ResourceManager:
    """Advanced resource management for parallel test execution"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.cpu_count = os.cpu_count()
        self.max_workers = max_workers or min(self.cpu_count, 8)
        self.total_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
        self.workers: Dict[str, WorkerResource] = {}
        self.resource_lock = threading.Lock()
        self.metrics: List[TestExecutionMetrics] = []
        
    def allocate_worker(self, worker_id: str) -> WorkerResource:
        """Allocate resources for a new worker"""
        with self.resource_lock:
            if worker_id in self.workers:
                return self.workers[worker_id]
            
            # Calculate optimal CPU affinity
            worker_index = len(self.workers)
            cpu_per_worker = max(1, self.cpu_count // self.max_workers)
            start_cpu = (worker_index * cpu_per_worker) % self.cpu_count
            cpu_affinity = [
                (start_cpu + i) % self.cpu_count 
                for i in range(cpu_per_worker)
            ]
            
            # Calculate memory limit per worker
            memory_per_worker = max(512, self.total_memory // self.max_workers)
            
            worker = WorkerResource(
                worker_id=worker_id,
                cpu_affinity=cpu_affinity,
                memory_limit=memory_per_worker
            )
            
            self.workers[worker_id] = worker
            return worker
    
    def set_cpu_affinity(self, worker_id: str) -> bool:
        """Set CPU affinity for the current process"""
        try:
            if worker_id not in self.workers:
                return False
                
            worker = self.workers[worker_id]
            process = psutil.Process()
            process.cpu_affinity(worker.cpu_affinity)
            logger.info(f"Set CPU affinity for worker {worker_id}: {worker.cpu_affinity}")
            return True
        except Exception as e:
            logger.error(f"Failed to set CPU affinity for worker {worker_id}: {e}")
            return False
    
    def monitor_resource_usage(self, worker_id: str) -> Dict[str, float]:
        """Monitor resource usage for a worker"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                "memory_mb": memory_info.rss / (1024 * 1024),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.rss / (self.total_memory * 1024 * 1024) * 100
            }
        except Exception as e:
            logger.error(f"Failed to monitor resources for worker {worker_id}: {e}")
            return {"memory_mb": 0, "cpu_percent": 0, "memory_percent": 0}
    
    def check_memory_limit(self, worker_id: str) -> bool:
        """Check if worker is within memory limits"""
        if worker_id not in self.workers:
            return True
            
        worker = self.workers[worker_id]
        usage = self.monitor_resource_usage(worker_id)
        
        return usage["memory_mb"] <= worker.memory_limit
    
    def get_optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on system resources"""
        # Consider CPU cores, memory, and current system load
        cpu_count = self.cpu_count
        memory_gb = self.total_memory / 1024
        
        # Adjust based on system load
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        load_factor = max(0.5, 1.0 - (load_avg / cpu_count))
        
        # Calculate optimal workers
        optimal_workers = min(
            cpu_count,
            int(memory_gb / 2),  # Assume 2GB per worker
            int(cpu_count * load_factor)
        )
        
        return max(1, optimal_workers)


class TestExecutionProfiler:
    """Profile test execution times and patterns"""
    
    def __init__(self):
        self.profile_data: Dict[str, List[float]] = {}
        self.test_dependencies: Dict[str, List[str]] = {}
        self.failure_rates: Dict[str, float] = {}
        self.profile_file = Path("test_execution_profile.json")
        self.load_profile()
    
    def load_profile(self):
        """Load existing profile data"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    self.profile_data = data.get("execution_times", {})
                    self.test_dependencies = data.get("dependencies", {})
                    self.failure_rates = data.get("failure_rates", {})
            except Exception as e:
                logger.error(f"Failed to load profile data: {e}")
    
    def save_profile(self):
        """Save profile data to file"""
        try:
            data = {
                "execution_times": self.profile_data,
                "dependencies": self.test_dependencies,
                "failure_rates": self.failure_rates,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.profile_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile data: {e}")
    
    def record_execution(self, test_name: str, duration: float, success: bool):
        """Record test execution data"""
        if test_name not in self.profile_data:
            self.profile_data[test_name] = []
        
        self.profile_data[test_name].append(duration)
        
        # Keep only last 100 executions
        if len(self.profile_data[test_name]) > 100:
            self.profile_data[test_name] = self.profile_data[test_name][-100:]
        
        # Update failure rate
        if test_name not in self.failure_rates:
            self.failure_rates[test_name] = 0.0
        
        # Exponential moving average for failure rate
        alpha = 0.1
        self.failure_rates[test_name] = (
            alpha * (0.0 if success else 1.0) + 
            (1 - alpha) * self.failure_rates[test_name]
        )
    
    def get_average_duration(self, test_name: str) -> float:
        """Get average execution time for a test"""
        if test_name not in self.profile_data or not self.profile_data[test_name]:
            return 10.0  # Default estimate
        
        durations = self.profile_data[test_name]
        return sum(durations) / len(durations)
    
    def get_test_priority(self, test_name: str) -> float:
        """Calculate test priority for scheduling"""
        # Higher priority = run earlier
        # Factors: failure rate (higher = higher priority), duration (longer = lower priority)
        
        failure_rate = self.failure_rates.get(test_name, 0.0)
        duration = self.get_average_duration(test_name)
        
        # Normalize duration (assume max 300 seconds)
        normalized_duration = min(duration / 300.0, 1.0)
        
        # Priority = failure_rate * 2 + (1 - normalized_duration)
        return failure_rate * 2.0 + (1.0 - normalized_duration)
    
    def get_optimal_test_order(self, test_names: List[str]) -> List[str]:
        """Get optimal test execution order"""
        # Sort by priority (descending) and then by estimated duration (ascending)
        def sort_key(test_name: str) -> Tuple[float, float]:
            priority = self.get_test_priority(test_name)
            duration = self.get_average_duration(test_name)
            return (-priority, duration)  # Negative priority for descending order
        
        return sorted(test_names, key=sort_key)


class TestDistributionStrategy:
    """Intelligent test distribution strategies"""
    
    def __init__(self, profiler: TestExecutionProfiler):
        self.profiler = profiler
    
    def distribute_by_loadfile(self, test_files: List[str], num_workers: int) -> List[List[str]]:
        """Distribute tests by file with load balancing"""
        # Sort files by estimated total execution time
        def file_weight(filename: str) -> float:
            # Estimate based on file size and historical data
            try:
                file_size = Path(filename).stat().st_size
                base_weight = file_size / 1024  # KB
                
                # Adjust based on historical execution times
                # This is a simplified heuristic
                return base_weight
            except (FileNotFoundError, IOError, OSError) as e:
                return 1.0
        
        sorted_files = sorted(test_files, key=file_weight, reverse=True)
        
        # Distribute using round-robin with weight consideration
        workers = [[] for _ in range(num_workers)]
        worker_loads = [0.0] * num_workers
        
        for test_file in sorted_files:
            # Find worker with minimum load
            min_load_idx = worker_loads.index(min(worker_loads))
            workers[min_load_idx].append(test_file)
            worker_loads[min_load_idx] += file_weight(test_file)
        
        return workers
    
    def distribute_by_loadscope(self, test_names: List[str], num_workers: int) -> List[List[str]]:
        """Distribute tests by scope with intelligent grouping"""
        # Group tests by module/class
        test_groups: Dict[str, List[str]] = {}
        
        for test_name in test_names:
            # Extract module/class from test name
            parts = test_name.split("::")
            if len(parts) >= 2:
                scope = "::".join(parts[:-1])
            else:
                scope = "default"
            
            if scope not in test_groups:
                test_groups[scope] = []
            test_groups[scope].append(test_name)
        
        # Sort groups by estimated total execution time
        def group_weight(group: List[str]) -> float:
            return sum(self.profiler.get_average_duration(test) for test in group)
        
        sorted_groups = sorted(test_groups.items(), key=lambda x: group_weight(x[1]), reverse=True)
        
        # Distribute groups to workers
        workers = [[] for _ in range(num_workers)]
        worker_loads = [0.0] * num_workers
        
        for scope, group in sorted_groups:
            # Find worker with minimum load
            min_load_idx = worker_loads.index(min(worker_loads))
            workers[min_load_idx].extend(group)
            worker_loads[min_load_idx] += group_weight(group)
        
        return workers
    
    def distribute_by_worksteal(self, test_names: List[str], num_workers: int) -> List[List[str]]:
        """Distribute tests with work stealing capability"""
        # Start with optimal ordering
        ordered_tests = self.profiler.get_optimal_test_order(test_names)
        
        # Initially distribute evenly
        chunk_size = len(ordered_tests) // num_workers
        workers = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            if i == num_workers - 1:
                # Last worker gets remaining tests
                end_idx = len(ordered_tests)
            else:
                end_idx = (i + 1) * chunk_size
            
            workers.append(ordered_tests[start_idx:end_idx])
        
        return workers


class ParallelTestExecutor:
    """Advanced parallel test executor with intelligent distribution"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.resource_manager = ResourceManager(max_workers)
        self.profiler = TestExecutionProfiler()
        self.strategy = TestDistributionStrategy(self.profiler)
        self.execution_metrics: List[TestExecutionMetrics] = []
        self.worker_health: Dict[str, Dict[str, Any]] = {}
    
    def execute_test_batch(self, test_names: List[str], worker_id: str) -> List[TestExecutionMetrics]:
        """Execute a batch of tests in a worker process"""
        # Set CPU affinity for this worker
        self.resource_manager.set_cpu_affinity(worker_id)
        
        metrics = []
        
        for test_name in test_names:
            start_time = time.time()
            
            try:
                # Check memory limits
                if not self.resource_manager.check_memory_limit(worker_id):
                    logger.warning(f"Worker {worker_id} exceeding memory limit")
                    break
                
                # Simulate test execution (replace with actual pytest execution)
                success = self._execute_single_test(test_name)
                
                duration = time.time() - start_time
                resource_usage = self.resource_manager.monitor_resource_usage(worker_id)
                
                metric = TestExecutionMetrics(
                    test_name=test_name,
                    duration=duration,
                    memory_usage=resource_usage["memory_mb"],
                    cpu_usage=resource_usage["cpu_percent"],
                    success=success,
                    worker_id=worker_id
                )
                
                metrics.append(metric)
                self.profiler.record_execution(test_name, duration, success)
                
            except Exception as e:
                logger.error(f"Test {test_name} failed in worker {worker_id}: {e}")
                duration = time.time() - start_time
                metric = TestExecutionMetrics(
                    test_name=test_name,
                    duration=duration,
                    memory_usage=0,
                    cpu_usage=0,
                    success=False,
                    worker_id=worker_id
                )
                metrics.append(metric)
                self.profiler.record_execution(test_name, duration, False)
        
        return metrics
    
    def _execute_single_test(self, test_name: str) -> bool:
        """Execute a single test (placeholder for actual implementation)"""
        # This would integrate with pytest execution
        # For now, simulate execution
        import random
        time.sleep(random.uniform(0.1, 2.0))  # Simulate test execution
        return random.random() > 0.1  # 90% success rate
    
    def run_parallel_tests(self, test_names: List[str], distribution_strategy: str = "loadscope") -> Dict[str, Any]:
        """Run tests in parallel with specified distribution strategy"""
        start_time = time.time()
        
        # Determine optimal worker count
        num_workers = self.resource_manager.get_optimal_worker_count()
        logger.info(f"Using {num_workers} workers for parallel execution")
        
        # Distribute tests based on strategy
        if distribution_strategy == "loadfile":
            test_distributions = self.strategy.distribute_by_loadfile(test_names, num_workers)
        elif distribution_strategy == "loadscope":
            test_distributions = self.strategy.distribute_by_loadscope(test_names, num_workers)
        elif distribution_strategy == "worksteal":
            test_distributions = self.strategy.distribute_by_worksteal(test_names, num_workers)
        else:
            raise ValueError(f"Unknown distribution strategy: {distribution_strategy}")
        
        # Execute tests in parallel
        all_metrics = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i, test_batch in enumerate(test_distributions):
                if not test_batch:
                    continue
                    
                worker_id = f"worker_{i}"
                self.resource_manager.allocate_worker(worker_id)
                
                future = executor.submit(self.execute_test_batch, test_batch, worker_id)
                futures.append((worker_id, future))
            
            # Collect results
            for worker_id, future in futures:
                try:
                    metrics = future.result(timeout=300)  # 5 minute timeout
                    all_metrics.extend(metrics)
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Save profiling data
        self.profiler.save_profile()
        
        # Generate execution report
        report = self._generate_execution_report(all_metrics, total_duration, num_workers)
        
        return report
    
    def _generate_execution_report(self, metrics: List[TestExecutionMetrics], 
                                 total_duration: float, num_workers: int) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        if not metrics:
            return {"error": "No test metrics collected"}
        
        total_tests = len(metrics)
        successful_tests = sum(1 for m in metrics if m.success)
        failed_tests = total_tests - successful_tests
        
        # Calculate performance metrics
        avg_duration = sum(m.duration for m in metrics) / total_tests
        max_duration = max(m.duration for m in metrics)
        min_duration = min(m.duration for m in metrics)
        
        avg_memory = sum(m.memory_usage for m in metrics) / total_tests
        max_memory = max(m.memory_usage for m in metrics)
        
        avg_cpu = sum(m.cpu_usage for m in metrics) / total_tests
        max_cpu = max(m.cpu_usage for m in metrics)
        
        # Worker performance
        worker_stats = {}
        for metric in metrics:
            worker_id = metric.worker_id
            if worker_id not in worker_stats:
                worker_stats[worker_id] = {
                    "tests_executed": 0,
                    "total_duration": 0,
                    "success_rate": 0,
                    "avg_memory": 0,
                    "avg_cpu": 0
                }
            
            stats = worker_stats[worker_id]
            stats["tests_executed"] += 1
            stats["total_duration"] += metric.duration
            stats["success_rate"] += (1 if metric.success else 0)
            stats["avg_memory"] += metric.memory_usage
            stats["avg_cpu"] += metric.cpu_usage
        
        # Finalize worker stats
        for worker_id, stats in worker_stats.items():
            if stats["tests_executed"] > 0:
                stats["success_rate"] /= stats["tests_executed"]
                stats["avg_memory"] /= stats["tests_executed"]
                stats["avg_cpu"] /= stats["tests_executed"]
        
        # Calculate speedup
        sequential_time = sum(m.duration for m in metrics)
        speedup = sequential_time / total_duration if total_duration > 0 else 0
        efficiency = speedup / num_workers if num_workers > 0 else 0
        
        report = {
            "execution_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "num_workers": num_workers,
                "speedup": speedup,
                "efficiency": efficiency
            },
            "performance_metrics": {
                "avg_test_duration": avg_duration,
                "max_test_duration": max_duration,
                "min_test_duration": min_duration,
                "avg_memory_usage": avg_memory,
                "max_memory_usage": max_memory,
                "avg_cpu_usage": avg_cpu,
                "max_cpu_usage": max_cpu
            },
            "worker_statistics": worker_stats,
            "slowest_tests": sorted(
                [(m.test_name, m.duration) for m in metrics],
                key=lambda x: x[1], reverse=True
            )[:10],
            "failed_tests": [m.test_name for m in metrics if not m.success],
            "timestamp": datetime.now().isoformat()
        }
        
        return report


# Integration with pytest-xdist
def pytest_configure_node(node):
    """Configure pytest worker node"""
    executor = ParallelTestExecutor()
    worker_id = getattr(node, 'workerid', 'master')
    
    if worker_id != 'master':
        executor.resource_manager.set_cpu_affinity(worker_id)


if __name__ == "__main__":
    # Demo execution
    executor = ParallelTestExecutor()
    
    # Mock test names
    test_names = [
        "tests/unit/test_config.py::test_config_loading",
        "tests/unit/test_event_bus.py::test_event_publishing",
        "tests/integration/test_strategic_marl.py::test_agent_coordination",
        "tests/performance/test_latency.py::test_response_time",
        "tests/risk/test_var_calculator.py::test_black_swan_scenarios",
        "tests/tactical/test_marl_system.py::test_concurrent_execution",
        "tests/security/test_authentication.py::test_token_validation",
        "tests/xai/test_explanation_engine.py::test_real_time_explanations"
    ]
    
    # Run with different strategies
    strategies = ["loadfile", "loadscope", "worksteal"]
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        report = executor.run_parallel_tests(test_names, strategy)
        
        print(f"Total tests: {report['execution_summary']['total_tests']}")
        print(f"Success rate: {report['execution_summary']['success_rate']:.2%}")
        print(f"Total duration: {report['execution_summary']['total_duration']:.2f}s")
        print(f"Speedup: {report['execution_summary']['speedup']:.2f}x")
        print(f"Efficiency: {report['execution_summary']['efficiency']:.2%}")