"""
Test Execution Profiling and Bottleneck Identification System
Agent 5 Mission: Real-time Test Monitoring & Analytics
"""

import cProfile
import pstats
import io
import time
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import structlog
import json
import sqlite3
from pathlib import Path
import line_profiler
import memory_profiler
import psutil
import sys
import traceback
from contextlib import contextmanager

logger = structlog.get_logger()


@dataclass
class ProfileResult:
    """Test execution profile result"""
    test_name: str
    profile_type: str  # cpu, memory, line, custom
    timestamp: datetime
    duration_ms: float
    total_calls: int
    hotspots: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    raw_data: Optional[str] = None


@dataclass
class BottleneckInfo:
    """Bottleneck information"""
    location: str  # function or line
    file_path: str
    line_number: int
    function_name: str
    time_ms: float
    percentage: float
    call_count: int
    severity: str  # low, medium, high, critical
    description: str
    suggested_fix: str


@dataclass
class TestExecutionProfile:
    """Comprehensive test execution profile"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    setup_time_ms: float
    teardown_time_ms: float
    test_body_time_ms: float
    cpu_profile: Optional[ProfileResult] = None
    memory_profile: Optional[ProfileResult] = None
    line_profile: Optional[ProfileResult] = None
    io_profile: Optional[Dict[str, Any]] = None
    network_profile: Optional[Dict[str, Any]] = None
    database_profile: Optional[Dict[str, Any]] = None
    custom_profiles: List[ProfileResult] = None
    
    def __post_init__(self):
        if self.custom_profiles is None:
            self.custom_profiles = []


class CPUProfiler:
    """CPU profiling for test execution"""
    
    def __init__(self):
        self.profiler = None
        self.profile_data = {}
    
    def start_profiling(self, test_name: str):
        """Start CPU profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profile_data[test_name] = {'start_time': time.time()}
    
    def stop_profiling(self, test_name: str) -> ProfileResult:
        """Stop CPU profiling and analyze results"""
        if not self.profiler:
            return None
        
        self.profiler.disable()
        end_time = time.time()
        
        # Convert profile to stats
        stats_stream = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=stats_stream)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        # Extract top functions
        hotspots = []
        bottlenecks = []
        
        for func_info in ps.get_stats_profile().func_profiles:
            if func_info.cumulative_time > 0.01:  # More than 10ms
                hotspot = {
                    'function': f"{func_info.file_name}:{func_info.line_number}({func_info.function_name})",
                    'cumulative_time': func_info.cumulative_time,
                    'self_time': func_info.self_time,
                    'call_count': func_info.call_count,
                    'percentage': (func_info.cumulative_time / (end_time - self.profile_data[test_name]['start_time'])) * 100
                }
                hotspots.append(hotspot)
                
                if hotspot['percentage'] > 5:  # More than 5% of total time
                    bottleneck = BottleneckInfo(
                        location=hotspot['function'],
                        file_path=func_info.file_name,
                        line_number=func_info.line_number,
                        function_name=func_info.function_name,
                        time_ms=func_info.cumulative_time * 1000,
                        percentage=hotspot['percentage'],
                        call_count=func_info.call_count,
                        severity='high' if hotspot['percentage'] > 20 else 'medium',
                        description=f"Function consuming {hotspot['percentage']:.1f}% of execution time",
                        suggested_fix=self._suggest_cpu_fix(func_info.function_name, hotspot['percentage'])
                    )
                    bottlenecks.append(asdict(bottleneck))
        
        # Sort by cumulative time
        hotspots.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        recommendations = self._generate_cpu_recommendations(hotspots)
        
        return ProfileResult(
            test_name=test_name,
            profile_type='cpu',
            timestamp=datetime.now(),
            duration_ms=(end_time - self.profile_data[test_name]['start_time']) * 1000,
            total_calls=sum(h['call_count'] for h in hotspots),
            hotspots=hotspots[:10],  # Top 10
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            raw_data=stats_stream.getvalue()
        )
    
    def _suggest_cpu_fix(self, function_name: str, percentage: float) -> str:
        """Suggest CPU optimization fix"""
        if percentage > 50:
            return "Critical bottleneck - consider algorithm optimization or caching"
        elif percentage > 20:
            return "Significant bottleneck - review implementation efficiency"
        elif percentage > 10:
            return "Minor bottleneck - consider micro-optimizations"
        else:
            return "Monitor for trends"
    
    def _generate_cpu_recommendations(self, hotspots: List[Dict[str, Any]]) -> List[str]:
        """Generate CPU optimization recommendations"""
        recommendations = []
        
        if not hotspots:
            return recommendations
        
        # Check for excessive function calls
        high_call_count = [h for h in hotspots if h['call_count'] > 10000]
        if high_call_count:
            recommendations.append("Consider reducing function call overhead or caching results")
        
        # Check for I/O operations in hot paths
        io_functions = [h for h in hotspots if any(keyword in h['function'].lower() 
                       for keyword in ['read', 'write', 'open', 'close', 'request'])]
        if io_functions:
            recommendations.append("I/O operations detected in hot paths - consider async operations")
        
        # Check for string operations
        string_ops = [h for h in hotspots if 'str' in h['function'].lower()]
        if len(string_ops) > 3:
            recommendations.append("Multiple string operations - consider string optimization")
        
        return recommendations


class MemoryProfiler:
    """Memory profiling for test execution"""
    
    def __init__(self):
        self.start_memory = None
        self.memory_snapshots = []
        self.process = psutil.Process()
    
    def start_profiling(self, test_name: str):
        """Start memory profiling"""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots = []
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'memory_mb': self.start_memory,
            'event': 'test_start'
        })
    
    def add_checkpoint(self, checkpoint_name: str):
        """Add memory checkpoint"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'memory_mb': current_memory,
            'event': checkpoint_name
        })
    
    def stop_profiling(self, test_name: str) -> ProfileResult:
        """Stop memory profiling and analyze results"""
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'memory_mb': end_memory,
            'event': 'test_end'
        })
        
        # Analyze memory usage
        memory_delta = end_memory - self.start_memory
        peak_memory = max(s['memory_mb'] for s in self.memory_snapshots)
        
        # Find memory growth patterns
        hotspots = []
        bottlenecks = []
        
        for i in range(1, len(self.memory_snapshots)):
            prev_snapshot = self.memory_snapshots[i-1]
            curr_snapshot = self.memory_snapshots[i]
            
            memory_growth = curr_snapshot['memory_mb'] - prev_snapshot['memory_mb']
            
            if memory_growth > 10:  # More than 10MB growth
                hotspot = {
                    'checkpoint': curr_snapshot['event'],
                    'memory_growth_mb': memory_growth,
                    'total_memory_mb': curr_snapshot['memory_mb'],
                    'timestamp': curr_snapshot['timestamp'].isoformat()
                }
                hotspots.append(hotspot)
                
                if memory_growth > 100:  # More than 100MB
                    bottleneck = BottleneckInfo(
                        location=curr_snapshot['event'],
                        file_path="unknown",
                        line_number=0,
                        function_name=curr_snapshot['event'],
                        time_ms=0,
                        percentage=(memory_growth / peak_memory) * 100,
                        call_count=1,
                        severity='high' if memory_growth > 500 else 'medium',
                        description=f"Memory growth of {memory_growth:.1f}MB at {curr_snapshot['event']}",
                        suggested_fix=self._suggest_memory_fix(memory_growth)
                    )
                    bottlenecks.append(asdict(bottleneck))
        
        recommendations = self._generate_memory_recommendations(memory_delta, peak_memory, hotspots)
        
        return ProfileResult(
            test_name=test_name,
            profile_type='memory',
            timestamp=datetime.now(),
            duration_ms=0,  # Memory profiling doesn't track time
            total_calls=len(self.memory_snapshots),
            hotspots=hotspots,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            raw_data=json.dumps(self.memory_snapshots, default=str)
        )
    
    def _suggest_memory_fix(self, memory_growth: float) -> str:
        """Suggest memory optimization fix"""
        if memory_growth > 500:
            return "Critical memory leak - investigate object lifecycle"
        elif memory_growth > 100:
            return "High memory usage - consider memory pooling or cleanup"
        elif memory_growth > 50:
            return "Moderate memory usage - review data structures"
        else:
            return "Monitor for memory leaks"
    
    def _generate_memory_recommendations(self, memory_delta: float, peak_memory: float, 
                                       hotspots: List[Dict[str, Any]]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if memory_delta > 100:
            recommendations.append("High memory usage detected - investigate memory leaks")
        
        if peak_memory > 1000:
            recommendations.append("Peak memory usage exceeded 1GB - consider memory optimization")
        
        if len(hotspots) > 5:
            recommendations.append("Multiple memory growth points - review test data management")
        
        return recommendations


class IOProfiler:
    """I/O profiling for test execution"""
    
    def __init__(self):
        self.start_io = None
        self.io_operations = []
        self.process = psutil.Process()
    
    def start_profiling(self, test_name: str):
        """Start I/O profiling"""
        try:
            self.start_io = self.process.io_counters()
            self.io_operations = []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.start_io = None
    
    def stop_profiling(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Stop I/O profiling and analyze results"""
        if self.start_io is None:
            return None
        
        try:
            end_io = self.process.io_counters()
            
            io_profile = {
                'read_bytes': end_io.read_bytes - self.start_io.read_bytes,
                'write_bytes': end_io.write_bytes - self.start_io.write_bytes,
                'read_count': end_io.read_count - self.start_io.read_count,
                'write_count': end_io.write_count - self.start_io.write_count,
                'read_mb': (end_io.read_bytes - self.start_io.read_bytes) / 1024 / 1024,
                'write_mb': (end_io.write_bytes - self.start_io.write_bytes) / 1024 / 1024
            }
            
            return io_profile
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class NetworkProfiler:
    """Network profiling for test execution"""
    
    def __init__(self):
        self.start_network = None
        self.network_operations = []
    
    def start_profiling(self, test_name: str):
        """Start network profiling"""
        try:
            self.start_network = psutil.net_io_counters()
            self.network_operations = []
        except Exception:
            self.start_network = None
    
    def stop_profiling(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Stop network profiling and analyze results"""
        if self.start_network is None:
            return None
        
        try:
            end_network = psutil.net_io_counters()
            
            network_profile = {
                'bytes_sent': end_network.bytes_sent - self.start_network.bytes_sent,
                'bytes_recv': end_network.bytes_recv - self.start_network.bytes_recv,
                'packets_sent': end_network.packets_sent - self.start_network.packets_sent,
                'packets_recv': end_network.packets_recv - self.start_network.packets_recv,
                'mb_sent': (end_network.bytes_sent - self.start_network.bytes_sent) / 1024 / 1024,
                'mb_recv': (end_network.bytes_recv - self.start_network.bytes_recv) / 1024 / 1024
            }
            
            return network_profile
        except Exception:
            return None


class DatabaseProfiler:
    """Database profiling for test execution"""
    
    def __init__(self):
        self.queries = []
        self.start_time = None
    
    def start_profiling(self, test_name: str):
        """Start database profiling"""
        self.queries = []
        self.start_time = time.time()
    
    def log_query(self, query: str, duration_ms: float, result_count: int = 0):
        """Log a database query"""
        self.queries.append({
            'query': query,
            'duration_ms': duration_ms,
            'result_count': result_count,
            'timestamp': datetime.now()
        })
    
    def stop_profiling(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Stop database profiling and analyze results"""
        if not self.queries:
            return None
        
        total_query_time = sum(q['duration_ms'] for q in self.queries)
        slow_queries = [q for q in self.queries if q['duration_ms'] > 100]  # Slower than 100ms
        
        db_profile = {
            'total_queries': len(self.queries),
            'total_query_time_ms': total_query_time,
            'slow_queries': len(slow_queries),
            'avg_query_time_ms': total_query_time / len(self.queries),
            'slowest_query': max(self.queries, key=lambda q: q['duration_ms']) if self.queries else None,
            'query_distribution': self._analyze_query_distribution()
        }
        
        return db_profile
    
    def _analyze_query_distribution(self) -> Dict[str, int]:
        """Analyze query type distribution"""
        distribution = defaultdict(int)
        
        for query in self.queries:
            query_type = query['query'].strip().split()[0].upper()
            distribution[query_type] += 1
        
        return dict(distribution)


class TestExecutionProfiler:
    """Main test execution profiler"""
    
    def __init__(self):
        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler()
        self.io_profiler = IOProfiler()
        self.network_profiler = NetworkProfiler()
        self.database_profiler = DatabaseProfiler()
        
        self.active_profiles = {}
        self.completed_profiles = deque(maxlen=100)
        self.profile_history = sqlite3.connect("test_profiles.db")
        self.setup_database()
    
    def setup_database(self):
        """Setup database for profile storage"""
        cursor = self.profile_history.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                profile_type TEXT,
                timestamp DATETIME,
                duration_ms REAL,
                total_calls INTEGER,
                hotspots TEXT,
                bottlenecks TEXT,
                recommendations TEXT,
                raw_data TEXT
            )
        ''')
        
        self.profile_history.commit()
    
    def start_profiling(self, test_name: str, profile_types: List[str] = None) -> str:
        """Start comprehensive profiling for a test"""
        if profile_types is None:
            profile_types = ['cpu', 'memory', 'io', 'network']
        
        profile_id = f"profile_{test_name}_{int(time.time() * 1000)}"
        
        profile = TestExecutionProfile(
            test_name=test_name,
            start_time=datetime.now(),
            end_time=None,
            total_duration_ms=0,
            setup_time_ms=0,
            teardown_time_ms=0,
            test_body_time_ms=0
        )
        
        self.active_profiles[profile_id] = profile
        
        # Start individual profilers
        if 'cpu' in profile_types:
            self.cpu_profiler.start_profiling(test_name)
        
        if 'memory' in profile_types:
            self.memory_profiler.start_profiling(test_name)
        
        if 'io' in profile_types:
            self.io_profiler.start_profiling(test_name)
        
        if 'network' in profile_types:
            self.network_profiler.start_profiling(test_name)
        
        if 'database' in profile_types:
            self.database_profiler.start_profiling(test_name)
        
        logger.info(f"Started profiling for test: {test_name}")
        return profile_id
    
    def add_checkpoint(self, profile_id: str, checkpoint_name: str):
        """Add profiling checkpoint"""
        if profile_id not in self.active_profiles:
            return
        
        self.memory_profiler.add_checkpoint(checkpoint_name)
        logger.debug(f"Added checkpoint: {checkpoint_name}")
    
    def stop_profiling(self, profile_id: str) -> TestExecutionProfile:
        """Stop profiling and generate comprehensive report"""
        if profile_id not in self.active_profiles:
            return None
        
        profile = self.active_profiles[profile_id]
        profile.end_time = datetime.now()
        profile.total_duration_ms = (profile.end_time - profile.start_time).total_seconds() * 1000
        
        # Stop individual profilers and collect results
        profile.cpu_profile = self.cpu_profiler.stop_profiling(profile.test_name)
        profile.memory_profile = self.memory_profiler.stop_profiling(profile.test_name)
        profile.io_profile = self.io_profiler.stop_profiling(profile.test_name)
        profile.network_profile = self.network_profiler.stop_profiling(profile.test_name)
        profile.database_profile = self.database_profiler.stop_profiling(profile.test_name)
        
        # Store in database
        self._store_profile(profile)
        
        # Move to completed profiles
        self.completed_profiles.append(profile)
        del self.active_profiles[profile_id]
        
        logger.info(f"Completed profiling for test: {profile.test_name}")
        return profile
    
    def _store_profile(self, profile: TestExecutionProfile):
        """Store profile in database"""
        cursor = self.profile_history.cursor()
        
        # Store each profile result
        for profile_result in [profile.cpu_profile, profile.memory_profile]:
            if profile_result:
                cursor.execute('''
                    INSERT INTO test_profiles 
                    (test_name, profile_type, timestamp, duration_ms, total_calls, 
                     hotspots, bottlenecks, recommendations, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile_result.test_name,
                    profile_result.profile_type,
                    profile_result.timestamp,
                    profile_result.duration_ms,
                    profile_result.total_calls,
                    json.dumps(profile_result.hotspots),
                    json.dumps(profile_result.bottlenecks),
                    json.dumps(profile_result.recommendations),
                    profile_result.raw_data
                ))
        
        self.profile_history.commit()
    
    def analyze_bottlenecks(self, profile: TestExecutionProfile) -> List[BottleneckInfo]:
        """Analyze bottlenecks across all profiling types"""
        all_bottlenecks = []
        
        # CPU bottlenecks
        if profile.cpu_profile and profile.cpu_profile.bottlenecks:
            all_bottlenecks.extend(profile.cpu_profile.bottlenecks)
        
        # Memory bottlenecks
        if profile.memory_profile and profile.memory_profile.bottlenecks:
            all_bottlenecks.extend(profile.memory_profile.bottlenecks)
        
        # Sort by severity and impact
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        all_bottlenecks.sort(key=lambda b: severity_order.get(b.get('severity', 'low'), 1), reverse=True)
        
        return all_bottlenecks
    
    def generate_optimization_report(self, profile: TestExecutionProfile) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        bottlenecks = self.analyze_bottlenecks(profile)
        
        # Aggregate recommendations
        all_recommendations = []
        if profile.cpu_profile:
            all_recommendations.extend(profile.cpu_profile.recommendations)
        if profile.memory_profile:
            all_recommendations.extend(profile.memory_profile.recommendations)
        
        # Performance summary
        performance_summary = {
            'total_duration_ms': profile.total_duration_ms,
            'bottleneck_count': len(bottlenecks),
            'critical_bottlenecks': len([b for b in bottlenecks if b.get('severity') == 'critical']),
            'high_bottlenecks': len([b for b in bottlenecks if b.get('severity') == 'high']),
            'optimization_potential': self._calculate_optimization_potential(bottlenecks)
        }
        
        # Resource usage summary
        resource_summary = {}
        if profile.io_profile:
            resource_summary['io'] = {
                'read_mb': profile.io_profile.get('read_mb', 0),
                'write_mb': profile.io_profile.get('write_mb', 0),
                'total_operations': profile.io_profile.get('read_count', 0) + profile.io_profile.get('write_count', 0)
            }
        
        if profile.network_profile:
            resource_summary['network'] = {
                'sent_mb': profile.network_profile.get('mb_sent', 0),
                'recv_mb': profile.network_profile.get('mb_recv', 0),
                'total_packets': profile.network_profile.get('packets_sent', 0) + profile.network_profile.get('packets_recv', 0)
            }
        
        if profile.database_profile:
            resource_summary['database'] = profile.database_profile
        
        return {
            'test_name': profile.test_name,
            'timestamp': profile.end_time,
            'performance_summary': performance_summary,
            'resource_summary': resource_summary,
            'bottlenecks': bottlenecks,
            'recommendations': list(set(all_recommendations)),  # Remove duplicates
            'optimization_priority': self._prioritize_optimizations(bottlenecks)
        }
    
    def _calculate_optimization_potential(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """Calculate optimization potential"""
        if not bottlenecks:
            return "low"
        
        critical_count = len([b for b in bottlenecks if b.get('severity') == 'critical'])
        high_count = len([b for b in bottlenecks if b.get('severity') == 'high'])
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"
    
    def _prioritize_optimizations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Prioritize optimization recommendations"""
        priorities = []
        
        # Critical issues first
        critical_bottlenecks = [b for b in bottlenecks if b.get('severity') == 'critical']
        if critical_bottlenecks:
            priorities.append("Address critical bottlenecks immediately")
        
        # High-impact issues
        high_bottlenecks = [b for b in bottlenecks if b.get('severity') == 'high']
        if high_bottlenecks:
            priorities.append("Optimize high-impact bottlenecks")
        
        # Pattern-based recommendations
        cpu_issues = [b for b in bottlenecks if 'cpu' in b.get('description', '').lower()]
        memory_issues = [b for b in bottlenecks if 'memory' in b.get('description', '').lower()]
        
        if len(cpu_issues) > len(memory_issues):
            priorities.append("Focus on CPU optimization")
        elif len(memory_issues) > len(cpu_issues):
            priorities.append("Focus on memory optimization")
        
        return priorities
    
    def get_historical_trends(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """Get historical performance trends"""
        cursor = self.profile_history.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT profile_type, timestamp, duration_ms, total_calls
            FROM test_profiles
            WHERE test_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (test_name, cutoff_date))
        
        results = cursor.fetchall()
        
        if not results:
            return {'message': 'No historical data available'}
        
        # Group by profile type
        trends = defaultdict(list)
        for profile_type, timestamp, duration_ms, total_calls in results:
            trends[profile_type].append({
                'timestamp': timestamp,
                'duration_ms': duration_ms,
                'total_calls': total_calls
            })
        
        # Calculate trend statistics
        trend_analysis = {}
        for profile_type, data in trends.items():
            durations = [d['duration_ms'] for d in data if d['duration_ms']]
            if durations:
                trend_analysis[profile_type] = {
                    'avg_duration_ms': np.mean(durations),
                    'trend_slope': self._calculate_trend_slope(data),
                    'data_points': len(durations),
                    'latest_duration_ms': durations[-1] if durations else 0
                }
        
        return {
            'test_name': test_name,
            'period_days': days,
            'trends': trend_analysis,
            'timestamp': datetime.now()
        }
    
    def _calculate_trend_slope(self, data: List[Dict[str, Any]]) -> float:
        """Calculate trend slope for performance data"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array([d['duration_ms'] for d in data if d['duration_ms']])
        
        if len(y) < 2:
            return 0.0
        
        slope, _ = np.polyfit(x[:len(y)], y, 1)
        return slope


# Context manager for easy profiling
@contextmanager
def profile_test(test_name: str, profiler: TestExecutionProfiler, profile_types: List[str] = None):
    """Context manager for easy test profiling"""
    profile_id = profiler.start_profiling(test_name, profile_types)
    try:
        yield profile_id
    finally:
        profile = profiler.stop_profiling(profile_id)
        if profile:
            report = profiler.generate_optimization_report(profile)
            logger.info(f"Profiling completed for {test_name}", extra={'report': report})