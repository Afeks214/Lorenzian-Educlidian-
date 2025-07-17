"""
Comprehensive Test Result Aggregation and Formatting System
Collects, processes, and formats test results from multiple sources
"""

import json
import yaml
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import re
import statistics
import subprocess
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import xml.etree.ElementTree as ET
import pytest
import coverage
from .advanced_test_reporting import TestResult, TestSuite, TestStatus, TestSeverity, TestReportGenerator


class TestAggregationConfig:
    """Configuration for test result aggregation"""
    
    def __init__(self, config_path: str = "configs/test_aggregation.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'sources': {
                'pytest': {
                    'enabled': True,
                    'results_path': 'test_results/pytest',
                    'formats': ['json', 'xml', 'html'],
                    'coverage_enabled': True
                },
                'unittest': {
                    'enabled': False,
                    'results_path': 'test_results/unittest',
                    'formats': ['xml']
                },
                'custom': {
                    'enabled': True,
                    'results_path': 'test_results/custom',
                    'formats': ['json']
                }
            },
            'aggregation': {
                'merge_duplicates': True,
                'timeout_threshold': 300,
                'memory_threshold': 1000,
                'failure_threshold': 0.05,
                'performance_baseline': 1.0
            },
            'formatting': {
                'timestamp_format': '%Y-%m-%d %H:%M:%S',
                'duration_precision': 3,
                'memory_unit': 'MB',
                'include_stack_trace': True,
                'max_error_length': 500
            },
            'output': {
                'formats': ['html', 'json', 'csv', 'junit'],
                'output_dir': 'test_reports',
                'archive_after_days': 30,
                'compress_archives': True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(user_config)
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value


class TestResultParser:
    """Parser for different test result formats"""
    
    def __init__(self, config: TestAggregationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def parse_pytest_json(self, file_path: str) -> List[TestResult]:
        """Parse pytest JSON results"""
        results = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for test_data in data.get('tests', []):
                result = TestResult(
                    test_id=test_data.get('nodeid', ''),
                    test_name=test_data.get('name', ''),
                    test_class=self._extract_class_name(test_data.get('nodeid', '')),
                    test_module=self._extract_module_name(test_data.get('nodeid', '')),
                    status=TestStatus(test_data.get('outcome', 'unknown')),
                    duration=test_data.get('duration', 0.0),
                    start_time=datetime.fromtimestamp(test_data.get('start', 0)),
                    end_time=datetime.fromtimestamp(test_data.get('end', 0)),
                    error_message=test_data.get('call', {}).get('longrepr', ''),
                    markers=test_data.get('markers', []),
                    parameters=test_data.get('parameters', {})
                )
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error parsing pytest JSON: {e}")
        
        return results
    
    def parse_junit_xml(self, file_path: str) -> List[TestResult]:
        """Parse JUnit XML results"""
        results = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for testcase in root.findall('.//testcase'):
                status = TestStatus.PASSED
                error_message = None
                stack_trace = None
                
                # Check for failures or errors
                failure = testcase.find('failure')
                error = testcase.find('error')
                skipped = testcase.find('skipped')
                
                if failure is not None:
                    status = TestStatus.FAILED
                    error_message = failure.get('message', '')
                    stack_trace = failure.text
                elif error is not None:
                    status = TestStatus.ERROR
                    error_message = error.get('message', '')
                    stack_trace = error.text
                elif skipped is not None:
                    status = TestStatus.SKIPPED
                    error_message = skipped.get('message', '')
                
                result = TestResult(
                    test_id=f"{testcase.get('classname')}.{testcase.get('name')}",
                    test_name=testcase.get('name', ''),
                    test_class=testcase.get('classname', ''),
                    test_module=self._extract_module_from_classname(testcase.get('classname', '')),
                    status=status,
                    duration=float(testcase.get('time', 0)),
                    start_time=datetime.now(),  # JUnit XML doesn't have start time
                    end_time=datetime.now(),
                    error_message=error_message,
                    stack_trace=stack_trace
                )
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error parsing JUnit XML: {e}")
        
        return results
    
    def parse_coverage_report(self, file_path: str) -> Dict[str, Any]:
        """Parse coverage report"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return {
                'total_coverage': data.get('totals', {}).get('percent_covered', 0),
                'line_coverage': data.get('totals', {}).get('percent_covered_display', '0%'),
                'branch_coverage': data.get('totals', {}).get('percent_covered', 0),
                'files': data.get('files', {}),
                'missing_lines': data.get('totals', {}).get('missing_lines', 0),
                'covered_lines': data.get('totals', {}).get('covered_lines', 0),
                'total_lines': data.get('totals', {}).get('num_statements', 0)
            }
        except Exception as e:
            self.logger.error(f"Error parsing coverage report: {e}")
            return {}
    
    def _extract_class_name(self, nodeid: str) -> str:
        """Extract class name from pytest nodeid"""
        match = re.search(r'::([^:]+)::test_', nodeid)
        return match.group(1) if match else ''
    
    def _extract_module_name(self, nodeid: str) -> str:
        """Extract module name from pytest nodeid"""
        if '::' in nodeid:
            return nodeid.split('::')[0].replace('/', '.').replace('.py', '')
        return nodeid.replace('/', '.').replace('.py', '')
    
    def _extract_module_from_classname(self, classname: str) -> str:
        """Extract module name from JUnit classname"""
        if '.' in classname:
            return '.'.join(classname.split('.')[:-1])
        return classname


class TestResultAggregator:
    """Aggregates test results from multiple sources"""
    
    def __init__(self, config: TestAggregationConfig):
        self.config = config
        self.parser = TestResultParser(config)
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config.get('output.output_dir', 'test_reports')) / 'aggregated_results.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize database for aggregated results"""
        self.db_path.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggregated_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_sources INTEGER,
                total_tests INTEGER,
                passed INTEGER,
                failed INTEGER,
                skipped INTEGER,
                errors INTEGER,
                duration REAL,
                coverage_percentage REAL,
                success_rate REAL,
                performance_score REAL,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggregated_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                test_id TEXT,
                test_name TEXT,
                test_class TEXT,
                test_module TEXT,
                status TEXT,
                duration REAL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                error_message TEXT,
                stack_trace TEXT,
                markers TEXT,
                parameters TEXT,
                severity TEXT,
                memory_usage REAL,
                cpu_usage REAL,
                source TEXT,
                reliability_score REAL,
                performance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES aggregated_runs (run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES aggregated_runs (run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def aggregate_results(self, source_directories: List[str]) -> TestSuite:
        """Aggregate test results from multiple sources"""
        all_results = []
        coverage_data = {}
        
        # Process sources concurrently
        tasks = []
        for source_dir in source_directories:
            tasks.append(self._process_source_directory(source_dir))
        
        source_results = await asyncio.gather(*tasks)
        
        # Combine all results
        for results, coverage in source_results:
            all_results.extend(results)
            if coverage:
                coverage_data.update(coverage)
        
        # Deduplicate and merge results
        merged_results = self._merge_duplicate_results(all_results)
        
        # Calculate metrics
        suite = self._create_test_suite(merged_results, coverage_data)
        
        # Store in database
        run_id = await self._store_aggregated_results(suite)
        
        # Calculate advanced metrics
        await self._calculate_advanced_metrics(suite, run_id)
        
        return suite
    
    async def _process_source_directory(self, source_dir: str) -> Tuple[List[TestResult], Dict[str, Any]]:
        """Process a single source directory"""
        source_path = Path(source_dir)
        results = []
        coverage_data = {}
        
        if not source_path.exists():
            self.logger.warning(f"Source directory does not exist: {source_dir}")
            return results, coverage_data
        
        # Process JSON files
        for json_file in source_path.glob('**/*.json'):
            if 'coverage' in json_file.name.lower():
                coverage_data = self.parser.parse_coverage_report(str(json_file))
            else:
                results.extend(self.parser.parse_pytest_json(str(json_file)))
        
        # Process XML files
        for xml_file in source_path.glob('**/*.xml'):
            results.extend(self.parser.parse_junit_xml(str(xml_file)))
        
        return results, coverage_data
    
    def _merge_duplicate_results(self, results: List[TestResult]) -> List[TestResult]:
        """Merge duplicate test results"""
        if not self.config.get('aggregation.merge_duplicates', True):
            return results
        
        test_map = {}
        for result in results:
            key = f"{result.test_module}.{result.test_class}.{result.test_name}"
            
            if key in test_map:
                # Merge with existing result
                existing = test_map[key]
                
                # Keep the most recent result
                if result.end_time > existing.end_time:
                    test_map[key] = result
                
                # Update duration (average)
                test_map[key].duration = (existing.duration + result.duration) / 2
            else:
                test_map[key] = result
        
        return list(test_map.values())
    
    def _create_test_suite(self, results: List[TestResult], coverage_data: Dict[str, Any]) -> TestSuite:
        """Create test suite from aggregated results"""
        if not results:
            return TestSuite(
                suite_name="Empty Suite",
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=0,
                total_duration=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                coverage_percentage=0.0,
                results=[]
            )
        
        # Calculate statistics
        total_tests = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        total_duration = sum(r.duration for r in results)
        start_time = min(r.start_time for r in results)
        end_time = max(r.end_time for r in results)
        
        coverage_percentage = coverage_data.get('total_coverage', 0.0)
        
        return TestSuite(
            suite_name=f"Aggregated_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total_duration=total_duration,
            start_time=start_time,
            end_time=end_time,
            coverage_percentage=coverage_percentage,
            results=results
        )
    
    async def _store_aggregated_results(self, suite: TestSuite) -> str:
        """Store aggregated results in database"""
        run_id = f"AGG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate advanced scores
        performance_score = self._calculate_performance_score(suite)
        quality_score = self._calculate_quality_score(suite)
        
        # Store run summary
        cursor.execute('''
            INSERT INTO aggregated_runs 
            (run_id, start_time, end_time, total_sources, total_tests, passed, failed, 
             skipped, errors, duration, coverage_percentage, success_rate, 
             performance_score, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, suite.start_time, suite.end_time, 1, suite.total_tests,
            suite.passed, suite.failed, suite.skipped, suite.errors,
            suite.total_duration, suite.coverage_percentage, suite.success_rate,
            performance_score, quality_score
        ))
        
        # Store individual results
        for result in suite.results:
            reliability_score = self._calculate_reliability_score(result)
            perf_score = self._calculate_test_performance_score(result)
            
            cursor.execute('''
                INSERT INTO aggregated_results 
                (run_id, test_id, test_name, test_class, test_module, status, duration,
                 start_time, end_time, error_message, stack_trace, markers, parameters,
                 severity, memory_usage, cpu_usage, source, reliability_score, performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, result.test_id, result.test_name, result.test_class,
                result.test_module, result.status.value, result.duration,
                result.start_time, result.end_time, result.error_message,
                result.stack_trace, json.dumps(result.markers),
                json.dumps(result.parameters), result.severity.value,
                result.memory_usage, result.cpu_usage, 'aggregated',
                reliability_score, perf_score
            ))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    async def _calculate_advanced_metrics(self, suite: TestSuite, run_id: str):
        """Calculate and store advanced metrics"""
        metrics = []
        
        # Performance metrics
        if suite.results:
            avg_duration = statistics.mean(r.duration for r in suite.results)
            median_duration = statistics.median(r.duration for r in suite.results)
            std_duration = statistics.stdev(r.duration for r in suite.results) if len(suite.results) > 1 else 0
            
            metrics.extend([
                ('avg_test_duration', avg_duration, 'performance'),
                ('median_test_duration', median_duration, 'performance'),
                ('std_test_duration', std_duration, 'performance'),
                ('max_test_duration', max(r.duration for r in suite.results), 'performance'),
                ('min_test_duration', min(r.duration for r in suite.results), 'performance')
            ])
        
        # Quality metrics
        flaky_tests = self._identify_flaky_tests(suite.results)
        slow_tests = sum(1 for r in suite.results if r.duration > self.config.get('aggregation.performance_baseline', 1.0))
        
        metrics.extend([
            ('flaky_test_count', len(flaky_tests), 'quality'),
            ('slow_test_count', slow_tests, 'quality'),
            ('test_density', suite.total_tests / max(1, len(set(r.test_module for r in suite.results))), 'quality'),
            ('error_rate', suite.errors / max(1, suite.total_tests), 'quality')
        ])
        
        # Coverage metrics
        metrics.extend([
            ('line_coverage', suite.coverage_percentage, 'coverage'),
            ('coverage_score', self._calculate_coverage_score(suite.coverage_percentage), 'coverage')
        ])
        
        # Store metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value, metric_type in metrics:
            cursor.execute('''
                INSERT INTO test_metrics (run_id, metric_name, metric_value, metric_type)
                VALUES (?, ?, ?, ?)
            ''', (run_id, metric_name, metric_value, metric_type))
        
        conn.commit()
        conn.close()
    
    def _calculate_performance_score(self, suite: TestSuite) -> float:
        """Calculate overall performance score"""
        if not suite.results:
            return 0.0
        
        avg_duration = statistics.mean(r.duration for r in suite.results)
        baseline = self.config.get('aggregation.performance_baseline', 1.0)
        
        # Score based on duration (lower is better)
        duration_score = max(0, 100 - (avg_duration / baseline * 100))
        
        # Score based on success rate
        success_score = suite.success_rate
        
        # Combined score
        return (duration_score * 0.3 + success_score * 0.7)
    
    def _calculate_quality_score(self, suite: TestSuite) -> float:
        """Calculate overall quality score"""
        if not suite.results:
            return 0.0
        
        # Base score from success rate
        base_score = suite.success_rate
        
        # Penalty for high error rate
        error_penalty = (suite.errors / suite.total_tests) * 50
        
        # Bonus for good coverage
        coverage_bonus = min(20, suite.coverage_percentage / 5)
        
        # Penalty for flaky tests
        flaky_penalty = len(self._identify_flaky_tests(suite.results)) * 5
        
        return max(0, min(100, base_score - error_penalty + coverage_bonus - flaky_penalty))
    
    def _calculate_reliability_score(self, result: TestResult) -> float:
        """Calculate reliability score for individual test"""
        base_score = 100.0
        
        # Penalty for failures
        if result.status == TestStatus.FAILED:
            base_score -= 50
        elif result.status == TestStatus.ERROR:
            base_score -= 70
        elif result.status == TestStatus.SKIPPED:
            base_score -= 10
        
        # Penalty for long duration
        if result.duration > self.config.get('aggregation.performance_baseline', 1.0):
            base_score -= min(30, (result.duration / 1.0) * 10)
        
        return max(0, base_score)
    
    def _calculate_test_performance_score(self, result: TestResult) -> float:
        """Calculate performance score for individual test"""
        baseline = self.config.get('aggregation.performance_baseline', 1.0)
        
        if result.duration <= baseline * 0.5:
            return 100.0
        elif result.duration <= baseline:
            return 80.0
        elif result.duration <= baseline * 2:
            return 60.0
        elif result.duration <= baseline * 5:
            return 40.0
        else:
            return 20.0
    
    def _calculate_coverage_score(self, coverage_percentage: float) -> float:
        """Calculate coverage score"""
        if coverage_percentage >= 95:
            return 100.0
        elif coverage_percentage >= 90:
            return 90.0
        elif coverage_percentage >= 80:
            return 80.0
        elif coverage_percentage >= 70:
            return 70.0
        else:
            return coverage_percentage
    
    def _identify_flaky_tests(self, results: List[TestResult]) -> List[TestResult]:
        """Identify potentially flaky tests"""
        # For now, identify tests that have inconsistent results
        # This would require historical data in a real implementation
        flaky_tests = []
        
        # Simple heuristic: tests with very variable durations
        test_durations = {}
        for result in results:
            key = f"{result.test_module}.{result.test_name}"
            if key not in test_durations:
                test_durations[key] = []
            test_durations[key].append(result.duration)
        
        for test_name, durations in test_durations.items():
            if len(durations) > 1:
                std_dev = statistics.stdev(durations)
                mean_duration = statistics.mean(durations)
                
                # If standard deviation is more than 50% of mean, consider it flaky
                if std_dev > (mean_duration * 0.5):
                    flaky_test = next((r for r in results if f"{r.test_module}.{r.test_name}" == test_name), None)
                    if flaky_test:
                        flaky_tests.append(flaky_test)
        
        return flaky_tests
    
    def generate_aggregated_report(self, run_id: str) -> Dict[str, Any]:
        """Generate comprehensive aggregated report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get run data
        run_data = pd.read_sql_query('''
            SELECT * FROM aggregated_runs WHERE run_id = ?
        ''', conn, params=(run_id,))
        
        # Get test results
        results_data = pd.read_sql_query('''
            SELECT * FROM aggregated_results WHERE run_id = ?
        ''', conn, params=(run_id,))
        
        # Get metrics
        metrics_data = pd.read_sql_query('''
            SELECT * FROM test_metrics WHERE run_id = ?
        ''', conn, params=(run_id,))
        
        conn.close()
        
        if run_data.empty:
            return {'error': 'Run not found'}
        
        # Build comprehensive report
        report = {
            'run_info': run_data.iloc[0].to_dict(),
            'summary': {
                'total_tests': int(run_data.iloc[0]['total_tests']),
                'passed': int(run_data.iloc[0]['passed']),
                'failed': int(run_data.iloc[0]['failed']),
                'skipped': int(run_data.iloc[0]['skipped']),
                'errors': int(run_data.iloc[0]['errors']),
                'success_rate': float(run_data.iloc[0]['success_rate']),
                'duration': float(run_data.iloc[0]['duration']),
                'coverage': float(run_data.iloc[0]['coverage_percentage']),
                'performance_score': float(run_data.iloc[0]['performance_score']),
                'quality_score': float(run_data.iloc[0]['quality_score'])
            },
            'test_results': results_data.to_dict('records'),
            'metrics': {
                row['metric_name']: {
                    'value': row['metric_value'],
                    'type': row['metric_type']
                }
                for _, row in metrics_data.iterrows()
            },
            'analysis': {
                'top_failures': self._get_top_failures(results_data),
                'slowest_tests': self._get_slowest_tests(results_data),
                'module_analysis': self._analyze_modules(results_data),
                'trend_indicators': self._calculate_trend_indicators(run_id)
            }
        }
        
        return report
    
    def _get_top_failures(self, results_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get top failure patterns"""
        failed_tests = results_data[results_data['status'] == 'failed']
        
        if failed_tests.empty:
            return []
        
        # Group by error message patterns
        error_patterns = failed_tests.groupby('error_message').size().sort_values(ascending=False)
        
        return [
            {
                'error_pattern': pattern,
                'count': count,
                'percentage': (count / len(failed_tests)) * 100
            }
            for pattern, count in error_patterns.head(10).items()
        ]
    
    def _get_slowest_tests(self, results_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get slowest tests"""
        slowest = results_data.nlargest(10, 'duration')
        
        return [
            {
                'test_name': row['test_name'],
                'module': row['test_module'],
                'duration': row['duration'],
                'status': row['status']
            }
            for _, row in slowest.iterrows()
        ]
    
    def _analyze_modules(self, results_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze test results by module"""
        module_stats = results_data.groupby('test_module').agg({
            'status': ['count', lambda x: (x == 'passed').sum(), lambda x: (x == 'failed').sum()],
            'duration': ['mean', 'max', 'sum'],
            'reliability_score': 'mean',
            'performance_score': 'mean'
        }).round(2)
        
        # Flatten column names
        module_stats.columns = [
            'total_tests', 'passed', 'failed', 'avg_duration', 'max_duration', 
            'total_duration', 'avg_reliability', 'avg_performance'
        ]
        
        # Calculate success rate
        module_stats['success_rate'] = (module_stats['passed'] / module_stats['total_tests']) * 100
        
        return module_stats.to_dict('index')
    
    def _calculate_trend_indicators(self, current_run_id: str) -> Dict[str, Any]:
        """Calculate trend indicators compared to previous runs"""
        conn = sqlite3.connect(self.db_path)
        
        # Get last 5 runs
        previous_runs = pd.read_sql_query('''
            SELECT * FROM aggregated_runs 
            WHERE run_id != ? 
            ORDER BY created_at DESC 
            LIMIT 5
        ''', conn, params=(current_run_id,))
        
        conn.close()
        
        if previous_runs.empty:
            return {'message': 'No previous runs for comparison'}
        
        current_run = pd.read_sql_query('''
            SELECT * FROM aggregated_runs WHERE run_id = ?
        ''', sqlite3.connect(self.db_path), params=(current_run_id,))
        
        if current_run.empty:
            return {'message': 'Current run not found'}
        
        current = current_run.iloc[0]
        previous_avg = previous_runs.mean()
        
        trends = {}
        for metric in ['success_rate', 'duration', 'coverage_percentage', 'performance_score', 'quality_score']:
            current_value = current[metric]
            previous_value = previous_avg[metric]
            
            change = ((current_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
            
            trends[metric] = {
                'current': current_value,
                'previous_avg': previous_value,
                'change_percent': change,
                'trend': 'improving' if change > 0 else 'declining' if change < 0 else 'stable'
            }
        
        return trends