"""
Test Execution Time Profiling System
Agent 2 Mission: Advanced Test Execution Optimization

This module provides comprehensive test execution profiling, analysis,
and optimization recommendations for maximum performance.
"""

import json
import time
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import threading
import os
import psutil

logger = logging.getLogger(__name__)


@dataclass
class TestProfile:
    """Comprehensive test execution profile"""
    test_name: str
    file_path: str
    duration: float
    memory_peak: float
    cpu_usage: float
    success: bool
    timestamp: datetime
    worker_id: Optional[str] = None
    test_type: str = "unit"  # unit, integration, performance, etc.
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class TestExecutionDatabase:
    """SQLite database for test execution history"""
    
    def __init__(self, db_path: str = "test_execution_history.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the SQLite database with required tables"""
        with self.lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS test_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_peak REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TEXT NOT NULL,
                    worker_id TEXT,
                    test_type TEXT DEFAULT 'unit',
                    dependencies TEXT
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS test_performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    avg_duration REAL NOT NULL,
                    duration_trend REAL NOT NULL,
                    failure_rate REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_test_name ON test_executions(test_name)
            ''')
            
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON test_executions(timestamp)
            ''')
            
            self.conn.commit()
    
    def record_execution(self, profile: TestProfile):
        """Record a test execution"""
        with self.lock:
            self.conn.execute('''
                INSERT INTO test_executions 
                (test_name, file_path, duration, memory_peak, cpu_usage, success, 
                 timestamp, worker_id, test_type, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.test_name,
                profile.file_path,
                profile.duration,
                profile.memory_peak,
                profile.cpu_usage,
                profile.success,
                profile.timestamp.isoformat(),
                profile.worker_id,
                profile.test_type,
                json.dumps(profile.dependencies)
            ))
            self.conn.commit()
    
    def get_test_history(self, test_name: str, limit: int = 100) -> List[TestProfile]:
        """Get execution history for a specific test"""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT * FROM test_executions 
                WHERE test_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (test_name, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append(TestProfile(
                    test_name=row[1],
                    file_path=row[2],
                    duration=row[3],
                    memory_peak=row[4],
                    cpu_usage=row[5],
                    success=bool(row[6]),
                    timestamp=datetime.fromisoformat(row[7]),
                    worker_id=row[8],
                    test_type=row[9],
                    dependencies=json.loads(row[10]) if row[10] else []
                ))
            return results
    
    def get_slowest_tests(self, limit: int = 20) -> List[Tuple[str, float]]:
        """Get the slowest tests based on recent executions"""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT test_name, AVG(duration) as avg_duration
                FROM test_executions 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY test_name
                ORDER BY avg_duration DESC
                LIMIT ?
            ''', (limit,))
            
            return [(row[0], row[1]) for row in cursor.fetchall()]
    
    def get_flaky_tests(self, min_runs: int = 5) -> List[Tuple[str, float]]:
        """Get tests with high failure rates"""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT test_name, 
                       (1.0 - AVG(CAST(success AS FLOAT))) as failure_rate,
                       COUNT(*) as run_count
                FROM test_executions 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY test_name
                HAVING run_count >= ?
                ORDER BY failure_rate DESC
            ''', (min_runs,))
            
            return [(row[0], row[1]) for row in cursor.fetchall() if row[1] > 0]
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Dict[str, float]]:
        """Get performance trends for tests"""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT test_name, duration, timestamp
                FROM test_executions 
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY test_name, timestamp
            '''.format(days))
            
            trends = defaultdict(list)
            for row in cursor.fetchall():
                trends[row[0]].append((row[1], row[2]))
            
            # Calculate trends
            result = {}
            for test_name, data in trends.items():
                if len(data) < 2:
                    continue
                
                durations = [d[0] for d in data]
                avg_duration = statistics.mean(durations)
                
                # Simple linear trend calculation
                n = len(durations)
                x_sum = sum(range(n))
                y_sum = sum(durations)
                xy_sum = sum(i * durations[i] for i in range(n))
                x2_sum = sum(i * i for i in range(n))
                
                slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
                
                result[test_name] = {
                    'avg_duration': avg_duration,
                    'trend_slope': slope,
                    'sample_count': n
                }
            
            return result
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()


class TestExecutionProfiler:
    """Comprehensive test execution profiler"""
    
    def __init__(self, db_path: str = "test_execution_history.db"):
        self.db = TestExecutionDatabase(db_path)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.system_baseline = self._get_system_baseline()
        
    def _get_system_baseline(self) -> Dict[str, float]:
        """Get system baseline metrics"""
        return {
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
    
    def start_profiling(self, test_name: str, file_path: str, 
                       test_type: str = "unit", dependencies: List[str] = None):
        """Start profiling a test execution"""
        if dependencies is None:
            dependencies = []
            
        profile_data = {
            'test_name': test_name,
            'file_path': file_path,
            'test_type': test_type,
            'dependencies': dependencies,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / (1024**2),  # MB
            'process': psutil.Process()
        }
        
        self.active_profiles[test_name] = profile_data
        logger.debug(f"Started profiling: {test_name}")
    
    def stop_profiling(self, test_name: str, success: bool = True, 
                      worker_id: Optional[str] = None) -> TestProfile:
        """Stop profiling and record the results"""
        if test_name not in self.active_profiles:
            raise ValueError(f"No active profile for test: {test_name}")
        
        profile_data = self.active_profiles.pop(test_name)
        
        # Calculate metrics
        duration = time.time() - profile_data['start_time']
        current_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        memory_peak = max(current_memory, profile_data['start_memory'])
        
        # Get CPU usage (approximate)
        try:
            cpu_usage = profile_data['process'].cpu_percent()
        except (FileNotFoundError, IOError, OSError) as e:
            cpu_usage = 0.0
        
        # Create profile
        profile = TestProfile(
            test_name=test_name,
            file_path=profile_data['file_path'],
            duration=duration,
            memory_peak=memory_peak,
            cpu_usage=cpu_usage,
            success=success,
            timestamp=datetime.now(),
            worker_id=worker_id,
            test_type=profile_data['test_type'],
            dependencies=profile_data['dependencies']
        )
        
        # Record in database
        self.db.record_execution(profile)
        
        logger.debug(f"Stopped profiling: {test_name} (duration: {duration:.2f}s)")
        return profile
    
    def get_test_optimization_recommendations(self, test_name: str) -> Dict[str, Any]:
        """Get optimization recommendations for a specific test"""
        history = self.db.get_test_history(test_name)
        
        if not history:
            return {"error": "No execution history found"}
        
        # Calculate statistics
        durations = [p.duration for p in history]
        memory_peaks = [p.memory_peak for p in history]
        cpu_usages = [p.cpu_usage for p in history]
        success_rate = sum(1 for p in history if p.success) / len(history)
        
        avg_duration = statistics.mean(durations)
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0
        avg_memory = statistics.mean(memory_peaks)
        avg_cpu = statistics.mean(cpu_usages)
        
        # Generate recommendations
        recommendations = []
        
        # Performance recommendations
        if avg_duration > 10.0:
            recommendations.append({
                'type': 'performance',
                'severity': 'high',
                'message': f'Test is slow (avg: {avg_duration:.2f}s). Consider optimization or parallel execution.',
                'suggestion': 'Break into smaller tests or optimize implementation'
            })
        
        if duration_std > avg_duration * 0.5:
            recommendations.append({
                'type': 'stability',
                'severity': 'medium',
                'message': f'Test duration is inconsistent (std: {duration_std:.2f}s)',
                'suggestion': 'Investigate external dependencies or resource contention'
            })
        
        # Memory recommendations
        if avg_memory > 500:  # > 500MB
            recommendations.append({
                'type': 'memory',
                'severity': 'medium',
                'message': f'Test uses significant memory (avg: {avg_memory:.1f}MB)',
                'suggestion': 'Consider memory optimization or isolation'
            })
        
        # Reliability recommendations
        if success_rate < 0.95:
            recommendations.append({
                'type': 'reliability',
                'severity': 'high',
                'message': f'Test is flaky (success rate: {success_rate:.1%})',
                'suggestion': 'Investigate and fix reliability issues'
            })
        
        return {
            'test_name': test_name,
            'statistics': {
                'avg_duration': avg_duration,
                'duration_std': duration_std,
                'avg_memory': avg_memory,
                'avg_cpu': avg_cpu,
                'success_rate': success_rate,
                'run_count': len(history)
            },
            'recommendations': recommendations,
            'classification': self._classify_test(avg_duration, avg_memory, success_rate)
        }
    
    def _classify_test(self, duration: float, memory: float, success_rate: float) -> str:
        """Classify test based on characteristics"""
        if duration > 30 or memory > 1000:
            return "heavy"
        elif duration > 10 or memory > 500:
            return "medium"
        elif success_rate < 0.9:
            return "flaky"
        else:
            return "light"
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        slowest_tests = self.db.get_slowest_tests(20)
        flaky_tests = self.db.get_flaky_tests(5)
        trends = self.db.get_performance_trends(30)
        
        # Calculate overall statistics
        total_tests = len(trends)
        improving_tests = sum(1 for t in trends.values() if t['trend_slope'] < 0)
        degrading_tests = sum(1 for t in trends.values() if t['trend_slope'] > 0)
        
        # Generate recommendations by category
        performance_recommendations = []
        reliability_recommendations = []
        resource_recommendations = []
        
        for test_name, duration in slowest_tests[:5]:
            performance_recommendations.append({
                'test': test_name,
                'issue': f'Slow execution: {duration:.2f}s average',
                'priority': 'high' if duration > 30 else 'medium',
                'action': 'Optimize or parallelize'
            })
        
        for test_name, failure_rate in flaky_tests[:5]:
            reliability_recommendations.append({
                'test': test_name,
                'issue': f'High failure rate: {failure_rate:.1%}',
                'priority': 'high',
                'action': 'Investigate and fix flakiness'
            })
        
        # System resource recommendations
        baseline = self.system_baseline
        if baseline['load_average'] > baseline['cpu_count'] * 0.8:
            resource_recommendations.append({
                'type': 'cpu',
                'issue': 'High system load detected',
                'priority': 'medium',
                'action': 'Consider reducing parallel workers'
            })
        
        if baseline['memory_available'] < baseline['memory_total'] * 0.2:
            resource_recommendations.append({
                'type': 'memory',
                'issue': 'Low available memory',
                'priority': 'high',
                'action': 'Optimize memory usage or add limits'
            })
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests_analyzed': total_tests,
                'improving_tests': improving_tests,
                'degrading_tests': degrading_tests,
                'slowest_test_duration': slowest_tests[0][1] if slowest_tests else 0,
                'flaky_tests_count': len(flaky_tests)
            },
            'slowest_tests': slowest_tests,
            'flaky_tests': flaky_tests,
            'performance_trends': trends,
            'recommendations': {
                'performance': performance_recommendations,
                'reliability': reliability_recommendations,
                'resources': resource_recommendations
            },
            'system_baseline': baseline
        }
    
    def export_profiles(self, output_path: str = "test_profiles.json"):
        """Export profiling data to JSON"""
        report = self.generate_optimization_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported profiling data to {output_path}")
    
    def close(self):
        """Close the profiler and database connections"""
        self.db.close()


# Context manager for easy profiling
class TestProfiler:
    """Context manager for test profiling"""
    
    def __init__(self, profiler: TestExecutionProfiler, test_name: str, 
                 file_path: str, test_type: str = "unit", 
                 dependencies: List[str] = None, worker_id: Optional[str] = None):
        self.profiler = profiler
        self.test_name = test_name
        self.file_path = file_path
        self.test_type = test_type
        self.dependencies = dependencies or []
        self.worker_id = worker_id
        self.success = True
    
    def __enter__(self):
        self.profiler.start_profiling(
            self.test_name, self.file_path, self.test_type, self.dependencies
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        self.profiler.stop_profiling(self.test_name, self.success, self.worker_id)


if __name__ == "__main__":
    # Demo usage
    profiler = TestExecutionProfiler()
    
    # Simulate some test executions
    test_names = [
        "test_fast_unit",
        "test_slow_integration", 
        "test_flaky_network",
        "test_memory_intensive"
    ]
    
    for i in range(10):
        for test_name in test_names:
            with TestProfiler(profiler, test_name, f"tests/{test_name}.py"):
                # Simulate test execution
                import random
                time.sleep(random.uniform(0.1, 3.0))
                
                # Simulate occasional failures
                if random.random() < 0.1:
                    raise Exception("Simulated test failure")
    
    # Generate optimization report
    report = profiler.generate_optimization_report()
    
    print("\n=== Test Optimization Report ===")
    print(f"Total tests analyzed: {report['summary']['total_tests_analyzed']}")
    print(f"Improving tests: {report['summary']['improving_tests']}")
    print(f"Degrading tests: {report['summary']['degrading_tests']}")
    
    print("\n--- Slowest Tests ---")
    for test_name, duration in report['slowest_tests'][:5]:
        print(f"{test_name}: {duration:.2f}s")
    
    print("\n--- Flaky Tests ---")
    for test_name, failure_rate in report['flaky_tests'][:5]:
        print(f"{test_name}: {failure_rate:.1%} failure rate")
    
    print("\n--- Performance Recommendations ---")
    for rec in report['recommendations']['performance']:
        print(f"- {rec['test']}: {rec['issue']} ({rec['priority']} priority)")
    
    # Export report
    profiler.export_profiles("test_optimization_report.json")
    
    profiler.close()