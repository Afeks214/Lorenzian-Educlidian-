"""
Advanced Test Reporting System
Comprehensive test result aggregation, formatting, and visualization
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import statistics
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAIL = "xfail"
    XPASS = "xpass"
    ERROR = "error"


class TestSeverity(Enum):
    """Test failure severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestResult:
    """Individual test result data structure"""
    test_id: str
    test_name: str
    test_class: str
    test_module: str
    status: TestStatus
    duration: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    markers: List[str] = None
    parameters: Dict[str, Any] = None
    severity: TestSeverity = TestSeverity.MEDIUM
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    def __post_init__(self):
        if self.markers is None:
            self.markers = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class TestSuite:
    """Test suite execution summary"""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    start_time: datetime
    end_time: datetime
    coverage_percentage: float
    results: List[TestResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return (self.passed / self.total_tests) * 100 if self.total_tests > 0 else 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        return (self.failed / self.total_tests) * 100 if self.total_tests > 0 else 0


class TestReportGenerator:
    """Advanced test report generator with multiple output formats"""
    
    def __init__(self, output_dir: str = "test_reports"):
        """Initialize the test report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.db_path = self.output_dir / "test_history.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for test history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for test history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_tests INTEGER,
                passed INTEGER,
                failed INTEGER,
                skipped INTEGER,
                errors INTEGER,
                duration REAL,
                coverage_percentage REAL,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                date TEXT,
                avg_duration REAL,
                success_rate REAL,
                execution_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_run_id(self, suite: TestSuite) -> str:
        """Generate unique run ID"""
        data = f"{suite.suite_name}_{suite.start_time.isoformat()}_{suite.total_tests}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def save_test_results(self, suite: TestSuite) -> str:
        """Save test results to database"""
        run_id = self.generate_run_id(suite)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save test run summary
        cursor.execute('''
            INSERT OR REPLACE INTO test_runs 
            (run_id, start_time, end_time, total_tests, passed, failed, skipped, errors,
             duration, coverage_percentage, success_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id, suite.start_time, suite.end_time, suite.total_tests,
            suite.passed, suite.failed, suite.skipped, suite.errors,
            suite.total_duration, suite.coverage_percentage, suite.success_rate
        ))
        
        # Save individual test results
        for result in suite.results:
            cursor.execute('''
                INSERT INTO test_results 
                (run_id, test_id, test_name, test_class, test_module, status, duration,
                 start_time, end_time, error_message, stack_trace, markers, parameters,
                 severity, memory_usage, cpu_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, result.test_id, result.test_name, result.test_class,
                result.test_module, result.status.value, result.duration,
                result.start_time, result.end_time, result.error_message,
                result.stack_trace, json.dumps(result.markers),
                json.dumps(result.parameters), result.severity.value,
                result.memory_usage, result.cpu_usage
            ))
        
        conn.commit()
        conn.close()
        return run_id
    
    def generate_html_report(self, suite: TestSuite) -> str:
        """Generate enhanced HTML report with visualizations"""
        run_id = self.save_test_results(suite)
        
        # Create visualizations
        charts = self._create_charts(suite)
        
        # Generate HTML content
        html_content = self._generate_html_template(suite, charts, run_id)
        
        # Save HTML report
        report_path = self.output_dir / f"test_report_{run_id}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def _create_charts(self, suite: TestSuite) -> Dict[str, str]:
        """Create interactive charts for test results"""
        charts = {}
        
        # Test status distribution pie chart
        status_counts = {
            'Passed': suite.passed,
            'Failed': suite.failed,
            'Skipped': suite.skipped,
            'Errors': suite.errors
        }
        
        fig_status = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Test Status Distribution",
            color_discrete_map={
                'Passed': '#28a745',
                'Failed': '#dc3545',
                'Skipped': '#ffc107',
                'Errors': '#6c757d'
            }
        )
        fig_status.update_layout(height=400)
        charts['status_pie'] = fig_status.to_html(include_plotlyjs='cdn')
        
        # Test duration histogram
        durations = [result.duration for result in suite.results]
        fig_duration = px.histogram(
            x=durations,
            nbins=20,
            title="Test Duration Distribution",
            labels={'x': 'Duration (seconds)', 'y': 'Count'}
        )
        fig_duration.update_layout(height=400)
        charts['duration_hist'] = fig_duration.to_html(include_plotlyjs='cdn')
        
        # Test results by module
        module_results = {}
        for result in suite.results:
            if result.test_module not in module_results:
                module_results[result.test_module] = {'passed': 0, 'failed': 0, 'skipped': 0}
            module_results[result.test_module][result.status.value] += 1
        
        modules = list(module_results.keys())
        passed_counts = [module_results[m].get('passed', 0) for m in modules]
        failed_counts = [module_results[m].get('failed', 0) for m in modules]
        skipped_counts = [module_results[m].get('skipped', 0) for m in modules]
        
        fig_module = go.Figure(data=[
            go.Bar(name='Passed', x=modules, y=passed_counts, marker_color='#28a745'),
            go.Bar(name='Failed', x=modules, y=failed_counts, marker_color='#dc3545'),
            go.Bar(name='Skipped', x=modules, y=skipped_counts, marker_color='#ffc107')
        ])
        fig_module.update_layout(
            title="Test Results by Module",
            barmode='stack',
            height=400,
            xaxis_tickangle=-45
        )
        charts['module_bar'] = fig_module.to_html(include_plotlyjs='cdn')
        
        # Performance metrics
        if any(result.memory_usage for result in suite.results):
            memory_usage = [result.memory_usage or 0 for result in suite.results]
            test_names = [result.test_name[:30] + '...' if len(result.test_name) > 30 
                         else result.test_name for result in suite.results]
            
            fig_memory = px.bar(
                x=test_names,
                y=memory_usage,
                title="Memory Usage by Test",
                labels={'x': 'Test Name', 'y': 'Memory Usage (MB)'}
            )
            fig_memory.update_layout(height=400, xaxis_tickangle=-45)
            charts['memory_bar'] = fig_memory.to_html(include_plotlyjs='cdn')
        
        return charts
    
    def _generate_html_template(self, suite: TestSuite, charts: Dict[str, str], run_id: str) -> str:
        """Generate HTML template with embedded charts"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Test Report - {suite.suite_name}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .test-passed {{ color: #28a745; }}
                .test-failed {{ color: #dc3545; }}
                .test-skipped {{ color: #ffc107; }}
                .test-error {{ color: #6c757d; }}
                .metric-card {{ 
                    border-left: 4px solid #007bff; 
                    padding: 20px; 
                    margin: 10px 0; 
                }}
                .chart-container {{ margin: 20px 0; }}
                .test-details {{ font-size: 12px; }}
                .expandable {{ cursor: pointer; }}
                .expandable:hover {{ background-color: #f8f9fa; }}
                .stack-trace {{ 
                    background-color: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 4px; 
                    font-family: monospace; 
                    font-size: 12px; 
                    white-space: pre-wrap; 
                }}
                .performance-indicator {{
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                    color: white;
                }}
                .perf-fast {{ background-color: #28a745; }}
                .perf-medium {{ background-color: #ffc107; }}
                .perf-slow {{ background-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <div class="row">
                    <div class="col-12">
                        <h1 class="text-center mb-4">Test Report: {suite.suite_name}</h1>
                        <div class="alert alert-info">
                            <strong>Run ID:</strong> {run_id}<br>
                            <strong>Generated:</strong> {datetime.now().isoformat()}<br>
                            <strong>Duration:</strong> {suite.total_duration:.2f} seconds<br>
                            <strong>Coverage:</strong> {suite.coverage_percentage:.1f}%
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card bg-success text-white">
                            <h3>{suite.passed}</h3>
                            <p>Passed Tests</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card bg-danger text-white">
                            <h3>{suite.failed}</h3>
                            <p>Failed Tests</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card bg-warning text-white">
                            <h3>{suite.skipped}</h3>
                            <p>Skipped Tests</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card bg-info text-white">
                            <h3>{suite.success_rate:.1f}%</h3>
                            <p>Success Rate</p>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            {charts.get('status_pie', '')}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            {charts.get('duration_hist', '')}
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="chart-container">
                            {charts.get('module_bar', '')}
                        </div>
                    </div>
                </div>
                
                {charts.get('memory_bar', '') and f'''
                <div class="row">
                    <div class="col-12">
                        <div class="chart-container">
                            {charts.get('memory_bar', '')}
                        </div>
                    </div>
                </div>
                ''' or ''}
                
                <div class="row">
                    <div class="col-12">
                        <h3>Detailed Test Results</h3>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Test Name</th>
                                        <th>Module</th>
                                        <th>Status</th>
                                        <th>Duration</th>
                                        <th>Performance</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {self._generate_test_rows(suite.results)}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Test Execution Timeline</h3>
                        <div class="timeline">
                            {self._generate_timeline(suite.results)}
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Performance Summary</h3>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5>Fastest Tests</h5>
                                        {self._generate_fastest_tests(suite.results)}
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5>Slowest Tests</h5>
                                        {self._generate_slowest_tests(suite.results)}
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h5>Memory Usage</h5>
                                        {self._generate_memory_summary(suite.results)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    // Add expandable functionality for stack traces
                    const expandables = document.querySelectorAll('.expandable');
                    expandables.forEach(el => {{
                        el.addEventListener('click', function() {{
                            const content = this.nextElementSibling;
                            content.style.display = content.style.display === 'none' ? 'block' : 'none';
                        }});
                    }});
                    
                    // Add filtering functionality
                    const statusFilter = document.getElementById('statusFilter');
                    if (statusFilter) {{
                        statusFilter.addEventListener('change', function() {{
                            const filter = this.value;
                            const rows = document.querySelectorAll('tbody tr');
                            rows.forEach(row => {{
                                const status = row.querySelector('.test-status').textContent.toLowerCase();
                                if (filter === 'all' || status === filter) {{
                                    row.style.display = '';
                                }} else {{
                                    row.style.display = 'none';
                                }}
                            }});
                        }});
                    }}
                }});
            </script>
        </body>
        </html>
        """
    
    def _generate_test_rows(self, results: List[TestResult]) -> str:
        """Generate HTML table rows for test results"""
        rows = []
        for result in results:
            status_class = f"test-{result.status.value}"
            perf_class = self._get_performance_class(result.duration)
            
            # Truncate test name if too long
            display_name = result.test_name[:50] + '...' if len(result.test_name) > 50 else result.test_name
            
            row = f"""
            <tr>
                <td>{display_name}</td>
                <td>{result.test_module}</td>
                <td><span class="test-status {status_class}">{result.status.value.title()}</span></td>
                <td>{result.duration:.3f}s</td>
                <td><span class="performance-indicator {perf_class}">{perf_class[5:].upper()}</span></td>
                <td>
                    {self._generate_test_details(result)}
                </td>
            </tr>
            """
            rows.append(row)
        
        return '\n'.join(rows)
    
    def _generate_test_details(self, result: TestResult) -> str:
        """Generate detailed test information"""
        details = []
        
        if result.error_message:
            details.append(f"""
            <div class="expandable">
                <strong>Error:</strong> {result.error_message[:100]}...
            </div>
            <div class="stack-trace" style="display: none;">
                {result.stack_trace or 'No stack trace available'}
            </div>
            """)
        
        if result.markers:
            details.append(f"<strong>Markers:</strong> {', '.join(result.markers)}")
        
        if result.memory_usage:
            details.append(f"<strong>Memory:</strong> {result.memory_usage:.1f} MB")
        
        if result.cpu_usage:
            details.append(f"<strong>CPU:</strong> {result.cpu_usage:.1f}%")
        
        return '<br>'.join(details)
    
    def _get_performance_class(self, duration: float) -> str:
        """Get performance class based on duration"""
        if duration < 0.1:
            return "perf-fast"
        elif duration < 1.0:
            return "perf-medium"
        else:
            return "perf-slow"
    
    def _generate_timeline(self, results: List[TestResult]) -> str:
        """Generate execution timeline"""
        # Sort by start time
        sorted_results = sorted(results, key=lambda x: x.start_time)
        
        timeline_items = []
        for result in sorted_results[:10]:  # Show first 10 for brevity
            status_class = f"test-{result.status.value}"
            timeline_items.append(f"""
            <div class="timeline-item">
                <span class="time">{result.start_time.strftime('%H:%M:%S')}</span>
                <span class="{status_class}">{result.test_name}</span>
                <span class="duration">({result.duration:.3f}s)</span>
            </div>
            """)
        
        return '\n'.join(timeline_items)
    
    def _generate_fastest_tests(self, results: List[TestResult]) -> str:
        """Generate fastest tests list"""
        fastest = sorted(results, key=lambda x: x.duration)[:5]
        items = []
        for result in fastest:
            items.append(f"""
            <div class="small">
                <strong>{result.test_name[:30]}</strong><br>
                {result.duration:.3f}s
            </div>
            """)
        return '\n'.join(items)
    
    def _generate_slowest_tests(self, results: List[TestResult]) -> str:
        """Generate slowest tests list"""
        slowest = sorted(results, key=lambda x: x.duration, reverse=True)[:5]
        items = []
        for result in slowest:
            items.append(f"""
            <div class="small">
                <strong>{result.test_name[:30]}</strong><br>
                {result.duration:.3f}s
            </div>
            """)
        return '\n'.join(items)
    
    def _generate_memory_summary(self, results: List[TestResult]) -> str:
        """Generate memory usage summary"""
        memory_results = [r for r in results if r.memory_usage]
        if not memory_results:
            return "<p>No memory data available</p>"
        
        avg_memory = statistics.mean(r.memory_usage for r in memory_results)
        max_memory = max(r.memory_usage for r in memory_results)
        
        return f"""
        <div class="small">
            <strong>Average:</strong> {avg_memory:.1f} MB<br>
            <strong>Peak:</strong> {max_memory:.1f} MB<br>
            <strong>Tests with data:</strong> {len(memory_results)}
        </div>
        """
    
    def generate_junit_xml(self, suite: TestSuite) -> str:
        """Generate JUnit XML report"""
        root = ET.Element('testsuite')
        root.set('name', suite.suite_name)
        root.set('tests', str(suite.total_tests))
        root.set('failures', str(suite.failed))
        root.set('errors', str(suite.errors))
        root.set('skipped', str(suite.skipped))
        root.set('time', str(suite.total_duration))
        root.set('timestamp', suite.start_time.isoformat())
        
        for result in suite.results:
            testcase = ET.SubElement(root, 'testcase')
            testcase.set('classname', result.test_class)
            testcase.set('name', result.test_name)
            testcase.set('time', str(result.duration))
            
            if result.status == TestStatus.FAILED:
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', result.error_message or 'Test failed')
                failure.text = result.stack_trace or 'No stack trace'
            elif result.status == TestStatus.ERROR:
                error = ET.SubElement(testcase, 'error')
                error.set('message', result.error_message or 'Test error')
                error.text = result.stack_trace or 'No stack trace'
            elif result.status == TestStatus.SKIPPED:
                skipped = ET.SubElement(testcase, 'skipped')
                skipped.set('message', 'Test skipped')
        
        # Save XML report
        xml_path = self.output_dir / f"junit_{suite.suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"JUnit XML report generated: {xml_path}")
        return str(xml_path)
    
    def generate_json_report(self, suite: TestSuite) -> str:
        """Generate JSON report"""
        report_data = {
            'suite_name': suite.suite_name,
            'summary': {
                'total_tests': suite.total_tests,
                'passed': suite.passed,
                'failed': suite.failed,
                'skipped': suite.skipped,
                'errors': suite.errors,
                'success_rate': suite.success_rate,
                'total_duration': suite.total_duration,
                'coverage_percentage': suite.coverage_percentage,
                'start_time': suite.start_time.isoformat(),
                'end_time': suite.end_time.isoformat()
            },
            'results': [asdict(result) for result in suite.results],
            'performance_metrics': {
                'avg_duration': statistics.mean(r.duration for r in suite.results),
                'median_duration': statistics.median(r.duration for r in suite.results),
                'max_duration': max(r.duration for r in suite.results),
                'min_duration': min(r.duration for r in suite.results),
                'slow_tests': [
                    asdict(r) for r in sorted(suite.results, key=lambda x: x.duration, reverse=True)[:10]
                ]
            },
            'failure_analysis': {
                'failed_tests': [asdict(r) for r in suite.results if r.status == TestStatus.FAILED],
                'error_patterns': self._analyze_error_patterns(suite.results)
            }
        }
        
        # Save JSON report
        json_path = self.output_dir / f"report_{suite.suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {json_path}")
        return str(json_path)
    
    def generate_csv_report(self, suite: TestSuite) -> str:
        """Generate CSV report"""
        csv_data = []
        for result in suite.results:
            csv_data.append({
                'test_name': result.test_name,
                'test_class': result.test_class,
                'test_module': result.test_module,
                'status': result.status.value,
                'duration': result.duration,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'error_message': result.error_message,
                'severity': result.severity.value,
                'memory_usage': result.memory_usage,
                'cpu_usage': result.cpu_usage,
                'markers': ','.join(result.markers)
            })
        
        # Save CSV report
        csv_path = self.output_dir / f"report_{suite.suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"CSV report generated: {csv_path}")
        return str(csv_path)
    
    def _analyze_error_patterns(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze error patterns in failed tests"""
        failed_results = [r for r in results if r.status == TestStatus.FAILED]
        
        if not failed_results:
            return {}
        
        error_types = {}
        for result in failed_results:
            if result.error_message:
                # Extract error type from message
                error_type = result.error_message.split(':')[0] if ':' in result.error_message else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_failures': len(failed_results),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            'failure_rate_by_module': self._calculate_failure_rate_by_module(failed_results)
        }
    
    def _calculate_failure_rate_by_module(self, failed_results: List[TestResult]) -> Dict[str, float]:
        """Calculate failure rate by module"""
        module_failures = {}
        for result in failed_results:
            module = result.test_module
            module_failures[module] = module_failures.get(module, 0) + 1
        
        return module_failures
    
    async def generate_all_reports(self, suite: TestSuite) -> Dict[str, str]:
        """Generate all report formats concurrently"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            html_future = loop.run_in_executor(executor, self.generate_html_report, suite)
            json_future = loop.run_in_executor(executor, self.generate_json_report, suite)
            csv_future = loop.run_in_executor(executor, self.generate_csv_report, suite)
            junit_future = loop.run_in_executor(executor, self.generate_junit_xml, suite)
            
            html_path = await html_future
            json_path = await json_future
            csv_path = await csv_future
            junit_path = await junit_future
        
        return {
            'html': html_path,
            'json': json_path,
            'csv': csv_path,
            'junit': junit_path
        }
    
    def get_test_history(self, test_name: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get test execution history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if test_name:
            cursor.execute('''
                SELECT * FROM test_results 
                WHERE test_name = ? AND created_at > datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days), (test_name,))
        else:
            cursor.execute('''
                SELECT * FROM test_runs 
                WHERE created_at > datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    def generate_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis for test performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trend data
        df_runs = pd.read_sql_query('''
            SELECT * FROM test_runs 
            WHERE created_at > datetime('now', '-{} days')
            ORDER BY created_at
        '''.format(days), conn)
        
        df_results = pd.read_sql_query('''
            SELECT * FROM test_results 
            WHERE created_at > datetime('now', '-{} days')
            ORDER BY created_at
        '''.format(days), conn)
        
        conn.close()
        
        if df_runs.empty:
            return {'message': 'No data available for trend analysis'}
        
        # Calculate trends
        trends = {
            'success_rate_trend': df_runs['success_rate'].tolist(),
            'duration_trend': df_runs['duration'].tolist(),
            'coverage_trend': df_runs['coverage_percentage'].tolist(),
            'failure_trend': df_runs['failed'].tolist(),
            'dates': df_runs['created_at'].tolist(),
            'avg_success_rate': df_runs['success_rate'].mean(),
            'avg_duration': df_runs['duration'].mean(),
            'avg_coverage': df_runs['coverage_percentage'].mean(),
            'trend_direction': self._calculate_trend_direction(df_runs['success_rate']),
            'most_unstable_tests': self._identify_unstable_tests(df_results)
        }
        
        return trends
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return 'stable'
        
        recent_avg = series.tail(5).mean()
        older_avg = series.head(5).mean()
        
        diff = recent_avg - older_avg
        if diff > 5:
            return 'improving'
        elif diff < -5:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_unstable_tests(self, df_results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify tests with high failure rate variation"""
        if df_results.empty:
            return []
        
        # Group by test name and calculate success rate variation
        test_stability = df_results.groupby('test_name').agg({
            'status': lambda x: (x == 'passed').sum() / len(x),
            'duration': ['mean', 'std'],
            'test_name': 'count'
        }).reset_index()
        
        # Flatten column names
        test_stability.columns = ['test_name', 'success_rate', 'avg_duration', 'duration_std', 'execution_count']
        
        # Filter tests with enough executions and calculate instability score
        stable_tests = test_stability[test_stability['execution_count'] >= 3].copy()
        stable_tests['instability_score'] = (
            (1 - stable_tests['success_rate']) * 0.7 +
            (stable_tests['duration_std'] / stable_tests['avg_duration']).fillna(0) * 0.3
        )
        
        # Return top 10 most unstable tests
        return stable_tests.nlargest(10, 'instability_score')[
            ['test_name', 'success_rate', 'avg_duration', 'execution_count', 'instability_score']
        ].to_dict('records')