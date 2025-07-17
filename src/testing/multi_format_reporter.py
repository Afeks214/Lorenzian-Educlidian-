"""
Multi-format Test Report Generation System
Generates test reports in multiple formats: HTML, JSON, CSV, PDF, JUnit XML
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import io
import base64
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import aiofiles
import logging
from dataclasses import dataclass, asdict
from jinja2 import Environment, FileSystemLoader, Template
import zipfile
import tarfile
from .advanced_test_reporting import TestResult, TestSuite, TestReportGenerator
from .coverage_analyzer import CoverageReport


class ReportFormat(Enum):
    """Supported report formats"""
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    JUNIT_XML = "junit_xml"
    MARKDOWN = "markdown"
    EXCEL = "excel"
    ARCHIVE = "archive"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    formats: List[str]
    output_dir: str
    template_dir: str
    include_charts: bool = True
    include_trends: bool = True
    include_coverage: bool = True
    include_performance: bool = True
    compress_output: bool = False
    email_recipients: List[str] = None
    slack_webhook: str = None
    custom_branding: Dict[str, str] = None
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.custom_branding is None:
            self.custom_branding = {}


class MultiFormatReporter:
    """Multi-format test report generator"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_dir = Path(config.template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create default templates
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default templates for different formats"""
        
        # HTML template
        html_template = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Test Report - {{ suite.suite_name }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .report-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; }
                .metric-card { transition: transform 0.2s; }
                .metric-card:hover { transform: translateY(-5px); }
                .status-passed { color: #28a745; }
                .status-failed { color: #dc3545; }
                .status-skipped { color: #ffc107; }
                .chart-container { height: 400px; margin: 20px 0; }
                .executive-summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .recommendation { border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; background: #f8f9fa; }
                .trend-up { color: #28a745; }
                .trend-down { color: #dc3545; }
                .trend-stable { color: #6c757d; }
                .coverage-bar { height: 20px; background: #e9ecef; border-radius: 10px; position: relative; }
                .coverage-fill { height: 100%; background: #28a745; border-radius: 10px; }
                .coverage-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="report-header text-center">
                <h1>{{ suite.suite_name }} Test Report</h1>
                <p class="lead">Generated on {{ suite.end_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                {% if config.custom_branding.company_name %}
                <p>{{ config.custom_branding.company_name }}</p>
                {% endif %}
            </div>
            
            <div class="container-fluid py-4">
                <!-- Executive Summary -->
                <div class="executive-summary">
                    <h2>Executive Summary</h2>
                    <div class="row">
                        <div class="col-md-8">
                            <p><strong>Test Execution:</strong> {{ suite.total_tests }} tests completed in {{ suite.total_duration|round(2) }} seconds</p>
                            <p><strong>Success Rate:</strong> {{ suite.success_rate|round(1) }}% ({{ suite.passed }} passed, {{ suite.failed }} failed, {{ suite.skipped }} skipped)</p>
                            <p><strong>Coverage:</strong> {{ suite.coverage_percentage|round(1) }}% of code covered by tests</p>
                            <p><strong>Overall Assessment:</strong> 
                                {% if suite.success_rate >= 95 %}
                                    <span class="badge bg-success">Excellent</span>
                                {% elif suite.success_rate >= 80 %}
                                    <span class="badge bg-primary">Good</span>
                                {% elif suite.success_rate >= 60 %}
                                    <span class="badge bg-warning">Needs Improvement</span>
                                {% else %}
                                    <span class="badge bg-danger">Critical</span>
                                {% endif %}
                            </p>
                        </div>
                        <div class="col-md-4 text-center">
                            <div class="coverage-bar">
                                <div class="coverage-fill" style="width: {{ suite.coverage_percentage }}%"></div>
                                <div class="coverage-text">{{ suite.coverage_percentage|round(1) }}%</div>
                            </div>
                            <small>Code Coverage</small>
                        </div>
                    </div>
                </div>
                
                <!-- Key Metrics -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card metric-card text-center">
                            <div class="card-body">
                                <h3 class="status-passed">{{ suite.passed }}</h3>
                                <p>Passed</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card text-center">
                            <div class="card-body">
                                <h3 class="status-failed">{{ suite.failed }}</h3>
                                <p>Failed</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card text-center">
                            <div class="card-body">
                                <h3 class="status-skipped">{{ suite.skipped }}</h3>
                                <p>Skipped</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card text-center">
                            <div class="card-body">
                                <h3>{{ suite.total_duration|round(2) }}s</h3>
                                <p>Duration</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Charts -->
                {% if config.include_charts %}
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <div id="statusChart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <div id="durationChart"></div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Detailed Results -->
                <div class="row">
                    <div class="col-12">
                        <h3>Detailed Test Results</h3>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Test Name</th>
                                        <th>Status</th>
                                        <th>Duration</th>
                                        <th>Module</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for result in suite.results %}
                                    <tr>
                                        <td>{{ result.test_name }}</td>
                                        <td><span class="status-{{ result.status.value }}">{{ result.status.value.title() }}</span></td>
                                        <td>{{ result.duration|round(3) }}s</td>
                                        <td>{{ result.test_module }}</td>
                                        <td>
                                            {% if result.error_message %}
                                            <details>
                                                <summary>Error Details</summary>
                                                <pre>{{ result.error_message }}</pre>
                                            </details>
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Recommendations -->
                {% if recommendations %}
                <div class="row">
                    <div class="col-12">
                        <h3>Recommendations</h3>
                        {% for rec in recommendations %}
                        <div class="recommendation">
                            <i class="fas fa-lightbulb"></i> {{ rec }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // Status distribution chart
                const statusData = {
                    values: [{{ suite.passed }}, {{ suite.failed }}, {{ suite.skipped }}],
                    labels: ['Passed', 'Failed', 'Skipped'],
                    type: 'pie',
                    hole: 0.3,
                    marker: {
                        colors: ['#28a745', '#dc3545', '#ffc107']
                    }
                };
                
                const statusLayout = {
                    title: 'Test Status Distribution',
                    height: 350
                };
                
                Plotly.newPlot('statusChart', [statusData], statusLayout);
                
                // Duration chart
                const durationData = {
                    x: {{ suite.results|map(attribute='test_name')|map('truncate', 20)|list|tojson }},
                    y: {{ suite.results|map(attribute='duration')|list|tojson }},
                    type: 'bar',
                    marker: {
                        color: '#007bff'
                    }
                };
                
                const durationLayout = {
                    title: 'Test Duration',
                    xaxis: { title: 'Test Name' },
                    yaxis: { title: 'Duration (seconds)' },
                    height: 350
                };
                
                Plotly.newPlot('durationChart', [durationData], durationLayout);
            </script>
        </body>
        </html>
        '''
        
        # Create template file
        template_path = self.template_dir / "test_report.html"
        if not template_path.exists():
            with open(template_path, 'w') as f:
                f.write(html_template)
        
        # Markdown template
        md_template = '''
# Test Report: {{ suite.suite_name }}

**Generated:** {{ suite.end_time.strftime('%Y-%m-%d %H:%M:%S') }}

## Executive Summary

- **Total Tests:** {{ suite.total_tests }}
- **Passed:** {{ suite.passed }}
- **Failed:** {{ suite.failed }}
- **Skipped:** {{ suite.skipped }}
- **Success Rate:** {{ suite.success_rate|round(1) }}%
- **Duration:** {{ suite.total_duration|round(2) }} seconds
- **Coverage:** {{ suite.coverage_percentage|round(1) }}%

## Test Results

| Test Name | Status | Duration | Module |
|-----------|--------|----------|--------|
{% for result in suite.results %}
| {{ result.test_name }} | {{ result.status.value.title() }} | {{ result.duration|round(3) }}s | {{ result.test_module }} |
{% endfor %}

## Failed Tests

{% for result in suite.results %}
{% if result.status.value == 'failed' %}
### {{ result.test_name }}

**Error:** {{ result.error_message or 'No error message' }}

```
{{ result.stack_trace or 'No stack trace available' }}
```

{% endif %}
{% endfor %}

## Recommendations

{% for rec in recommendations %}
- {{ rec }}
{% endfor %}

*Report generated by Advanced Test Reporting System*
        '''
        
        md_template_path = self.template_dir / "test_report.md"
        if not md_template_path.exists():
            with open(md_template_path, 'w') as f:
                f.write(md_template)
    
    async def generate_reports(self, 
                              suite: TestSuite, 
                              coverage_report: Optional[CoverageReport] = None,
                              previous_results: Optional[List[TestSuite]] = None) -> Dict[str, str]:
        """Generate reports in all configured formats"""
        
        reports = {}
        
        # Generate base data
        report_data = self._prepare_report_data(suite, coverage_report, previous_results)
        
        # Generate charts if enabled
        if self.config.include_charts:
            chart_paths = await self._generate_charts(suite)
            report_data['charts'] = chart_paths
        
        # Generate each format
        for format_name in self.config.formats:
            try:
                if format_name == ReportFormat.HTML.value:
                    reports[format_name] = await self._generate_html_report(suite, report_data)
                elif format_name == ReportFormat.JSON.value:
                    reports[format_name] = await self._generate_json_report(suite, report_data)
                elif format_name == ReportFormat.CSV.value:
                    reports[format_name] = await self._generate_csv_report(suite, report_data)
                elif format_name == ReportFormat.PDF.value:
                    reports[format_name] = await self._generate_pdf_report(suite, report_data)
                elif format_name == ReportFormat.JUNIT_XML.value:
                    reports[format_name] = await self._generate_junit_xml_report(suite, report_data)
                elif format_name == ReportFormat.MARKDOWN.value:
                    reports[format_name] = await self._generate_markdown_report(suite, report_data)
                elif format_name == ReportFormat.EXCEL.value:
                    reports[format_name] = await self._generate_excel_report(suite, report_data)
                
            except Exception as e:
                self.logger.error(f"Error generating {format_name} report: {e}")
                continue
        
        # Create archive if requested
        if self.config.compress_output or ReportFormat.ARCHIVE.value in self.config.formats:
            reports['archive'] = await self._create_archive(reports)
        
        return reports
    
    def _prepare_report_data(self, 
                            suite: TestSuite, 
                            coverage_report: Optional[CoverageReport], 
                            previous_results: Optional[List[TestSuite]]) -> Dict[str, Any]:
        """Prepare comprehensive report data"""
        
        data = {
            'suite': suite,
            'coverage': coverage_report,
            'previous_results': previous_results or [],
            'config': self.config,
            'generated_at': datetime.now(),
            'recommendations': self._generate_recommendations(suite, coverage_report),
            'performance_metrics': self._calculate_performance_metrics(suite),
            'quality_metrics': self._calculate_quality_metrics(suite, coverage_report),
            'trends': self._calculate_trends(suite, previous_results) if previous_results else {}
        }
        
        return data
    
    def _generate_recommendations(self, suite: TestSuite, coverage_report: Optional[CoverageReport]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Test results recommendations
        if suite.success_rate < 80:
            recommendations.append("Test success rate is below 80%. Investigate failing tests and improve test stability.")
        
        if suite.failed > 0:
            recommendations.append(f"Address {suite.failed} failing tests to improve overall quality.")
        
        if suite.skipped > 0:
            recommendations.append(f"Review {suite.skipped} skipped tests - they may indicate missing functionality or configuration issues.")
        
        # Performance recommendations
        slow_tests = [r for r in suite.results if r.duration > 5.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow tests (>5s) to improve CI/CD performance.")
        
        # Coverage recommendations
        if coverage_report:
            if coverage_report.overall_coverage < 80:
                recommendations.append("Increase code coverage to at least 80% for better test confidence.")
            
            if coverage_report.uncovered_hotspots:
                recommendations.append("Focus on testing uncovered critical code paths identified in hotspots.")
        
        # General recommendations
        if len(suite.results) < 10:
            recommendations.append("Consider adding more tests to improve coverage and confidence.")
        
        return recommendations
    
    def _calculate_performance_metrics(self, suite: TestSuite) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not suite.results:
            return {}
        
        durations = [r.duration for r in suite.results]
        
        return {
            'avg_duration': sum(durations) / len(durations),
            'median_duration': sorted(durations)[len(durations) // 2],
            'max_duration': max(durations),
            'min_duration': min(durations),
            'total_duration': suite.total_duration,
            'tests_per_second': len(suite.results) / suite.total_duration if suite.total_duration > 0 else 0,
            'slow_tests': len([d for d in durations if d > 5.0]),
            'fast_tests': len([d for d in durations if d < 0.1])
        }
    
    def _calculate_quality_metrics(self, suite: TestSuite, coverage_report: Optional[CoverageReport]) -> Dict[str, Any]:
        """Calculate quality metrics"""
        metrics = {
            'success_rate': suite.success_rate,
            'failure_rate': suite.failure_rate,
            'test_density': len(suite.results) / max(1, len(set(r.test_module for r in suite.results))),
            'error_rate': suite.errors / suite.total_tests if suite.total_tests > 0 else 0
        }
        
        if coverage_report:
            metrics.update({
                'coverage_score': coverage_report.overall_coverage,
                'coverage_quality': self._assess_coverage_quality(coverage_report.overall_coverage)
            })
        
        return metrics
    
    def _assess_coverage_quality(self, coverage: float) -> str:
        """Assess coverage quality level"""
        if coverage >= 95:
            return "Excellent"
        elif coverage >= 80:
            return "Good"
        elif coverage >= 60:
            return "Fair"
        elif coverage >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def _calculate_trends(self, suite: TestSuite, previous_results: List[TestSuite]) -> Dict[str, Any]:
        """Calculate trends from previous results"""
        if not previous_results:
            return {}
        
        # Calculate trend indicators
        success_rates = [r.success_rate for r in previous_results] + [suite.success_rate]
        durations = [r.total_duration for r in previous_results] + [suite.total_duration]
        
        return {
            'success_rate_trend': self._calculate_trend_direction(success_rates),
            'duration_trend': self._calculate_trend_direction(durations, reverse=True),
            'test_count_trend': self._calculate_trend_direction([r.total_tests for r in previous_results] + [suite.total_tests]),
            'historical_success_rates': success_rates,
            'historical_durations': durations
        }
    
    def _calculate_trend_direction(self, values: List[float], reverse: bool = False) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent_avg = sum(values[-3:]) / len(values[-3:])
        older_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else values[0]
        
        diff = recent_avg - older_avg
        
        if reverse:
            diff = -diff
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    async def _generate_charts(self, suite: TestSuite) -> Dict[str, str]:
        """Generate charts for the report"""
        charts = {}
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        # Status distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [suite.passed, suite.failed, suite.skipped]
        labels = ['Passed', 'Failed', 'Skipped']
        colors = ['#28a745', '#dc3545', '#ffc107']
        
        # Filter out zero values
        non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
        if non_zero_data:
            sizes, labels, colors = zip(*non_zero_data)
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Test Status Distribution')
            
            status_chart_path = chart_dir / "status_distribution.png"
            plt.savefig(status_chart_path, dpi=300, bbox_inches='tight')
            charts['status_distribution'] = str(status_chart_path)
        
        plt.close()
        
        # Duration histogram
        if suite.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            durations = [r.duration for r in suite.results]
            ax.hist(durations, bins=20, color='#007bff', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Number of Tests')
            ax.set_title('Test Duration Distribution')
            
            duration_chart_path = chart_dir / "duration_distribution.png"
            plt.savefig(duration_chart_path, dpi=300, bbox_inches='tight')
            charts['duration_distribution'] = str(duration_chart_path)
            
            plt.close()
        
        # Module performance chart
        if suite.results:
            module_data = {}
            for result in suite.results:
                module = result.test_module
                if module not in module_data:
                    module_data[module] = {'count': 0, 'total_duration': 0, 'passed': 0, 'failed': 0}
                
                module_data[module]['count'] += 1
                module_data[module]['total_duration'] += result.duration
                
                if result.status.value == 'passed':
                    module_data[module]['passed'] += 1
                elif result.status.value == 'failed':
                    module_data[module]['failed'] += 1
            
            modules = list(module_data.keys())
            success_rates = [(data['passed'] / data['count']) * 100 for data in module_data.values()]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(modules, success_rates, color='#007bff', alpha=0.7)
            ax.set_xlabel('Module')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rate by Module')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            
            module_chart_path = chart_dir / "module_performance.png"
            plt.savefig(module_chart_path, dpi=300, bbox_inches='tight')
            charts['module_performance'] = str(module_chart_path)
            
            plt.close()
        
        return charts
    
    async def _generate_html_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        template = self.jinja_env.get_template('test_report.html')
        
        html_content = template.render(**data)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = self.output_dir / f"test_report_{timestamp}.html"
        
        async with aiofiles.open(html_path, 'w') as f:
            await f.write(html_content)
        
        return str(html_path)
    
    async def _generate_json_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate JSON report"""
        
        json_data = {
            'metadata': {
                'suite_name': suite.suite_name,
                'generated_at': data['generated_at'].isoformat(),
                'total_tests': suite.total_tests,
                'duration': suite.total_duration
            },
            'summary': {
                'passed': suite.passed,
                'failed': suite.failed,
                'skipped': suite.skipped,
                'errors': suite.errors,
                'success_rate': suite.success_rate,
                'failure_rate': suite.failure_rate,
                'coverage_percentage': suite.coverage_percentage
            },
            'results': [
                {
                    'test_name': result.test_name,
                    'test_module': result.test_module,
                    'test_class': result.test_class,
                    'status': result.status.value,
                    'duration': result.duration,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'error_message': result.error_message,
                    'stack_trace': result.stack_trace,
                    'markers': result.markers,
                    'parameters': result.parameters
                }
                for result in suite.results
            ],
            'performance_metrics': data['performance_metrics'],
            'quality_metrics': data['quality_metrics'],
            'recommendations': data['recommendations'],
            'trends': data['trends']
        }
        
        if data['coverage']:
            json_data['coverage'] = {
                'overall_coverage': data['coverage'].overall_coverage,
                'line_coverage': data['coverage'].line_coverage,
                'branch_coverage': data['coverage'].branch_coverage,
                'function_coverage': data['coverage'].function_coverage
            }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"test_report_{timestamp}.json"
        
        async with aiofiles.open(json_path, 'w') as f:
            await f.write(json.dumps(json_data, indent=2, default=str))
        
        return str(json_path)
    
    async def _generate_csv_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate CSV report"""
        
        csv_data = []
        for result in suite.results:
            csv_data.append({
                'test_name': result.test_name,
                'test_module': result.test_module,
                'test_class': result.test_class,
                'status': result.status.value,
                'duration': result.duration,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'error_message': result.error_message or '',
                'markers': ','.join(result.markers) if result.markers else '',
                'severity': result.severity.value if hasattr(result, 'severity') else '',
                'memory_usage': result.memory_usage or 0,
                'cpu_usage': result.cpu_usage or 0
            })
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f"test_report_{timestamp}.csv"
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    async def _generate_pdf_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate PDF report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = self.output_dir / f"test_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e')
        )
        
        # Title
        story.append(Paragraph(f"Test Report: {suite.suite_name}", title_style))
        story.append(Paragraph(f"Generated: {data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tests', str(suite.total_tests)],
            ['Passed', str(suite.passed)],
            ['Failed', str(suite.failed)],
            ['Skipped', str(suite.skipped)],
            ['Success Rate', f"{suite.success_rate:.1f}%"],
            ['Duration', f"{suite.total_duration:.2f} seconds"],
            ['Coverage', f"{suite.coverage_percentage:.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", heading_style))
        
        perf_metrics = data['performance_metrics']
        perf_data = [
            ['Metric', 'Value'],
            ['Average Duration', f"{perf_metrics.get('avg_duration', 0):.3f}s"],
            ['Median Duration', f"{perf_metrics.get('median_duration', 0):.3f}s"],
            ['Max Duration', f"{perf_metrics.get('max_duration', 0):.3f}s"],
            ['Min Duration', f"{perf_metrics.get('min_duration', 0):.3f}s"],
            ['Tests per Second', f"{perf_metrics.get('tests_per_second', 0):.2f}"],
            ['Slow Tests (>5s)', str(perf_metrics.get('slow_tests', 0))],
            ['Fast Tests (<0.1s)', str(perf_metrics.get('fast_tests', 0))]
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 2*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        # Failed Tests Details
        failed_tests = [result for result in suite.results if result.status.value == 'failed']
        if failed_tests:
            story.append(Paragraph("Failed Tests", heading_style))
            
            for test in failed_tests[:10]:  # Limit to first 10 failed tests
                story.append(Paragraph(f"<b>{test.test_name}</b>", styles['Normal']))
                story.append(Paragraph(f"Module: {test.test_module}", styles['Normal']))
                story.append(Paragraph(f"Duration: {test.duration:.3f}s", styles['Normal']))
                if test.error_message:
                    error_text = test.error_message[:200] + "..." if len(test.error_message) > 200 else test.error_message
                    story.append(Paragraph(f"Error: {error_text}", styles['Normal']))
                story.append(Spacer(1, 10))
            
            story.append(PageBreak())
        
        # Recommendations
        if data['recommendations']:
            story.append(Paragraph("Recommendations", heading_style))
            
            for i, rec in enumerate(data['recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 8))
        
        # Charts (if available)
        if 'charts' in data and data['charts']:
            story.append(PageBreak())
            story.append(Paragraph("Charts", heading_style))
            
            for chart_name, chart_path in data['charts'].items():
                if Path(chart_path).exists():
                    try:
                        story.append(Paragraph(chart_name.replace('_', ' ').title(), styles['Heading2']))
                        
                        # Resize image to fit page
                        img = Image(chart_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 20))
                    except Exception as e:
                        self.logger.warning(f"Could not include chart {chart_name}: {e}")
        
        # Build PDF
        doc.build(story)
        
        return str(pdf_path)
    
    async def _generate_junit_xml_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
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
            testcase.set('classname', result.test_class or result.test_module)
            testcase.set('name', result.test_name)
            testcase.set('time', str(result.duration))
            
            if result.status.value == 'failed':
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', result.error_message or 'Test failed')
                failure.text = result.stack_trace or 'No stack trace available'
            elif result.status.value == 'error':
                error = ET.SubElement(testcase, 'error')
                error.set('message', result.error_message or 'Test error')
                error.text = result.stack_trace or 'No stack trace available'
            elif result.status.value == 'skipped':
                skipped = ET.SubElement(testcase, 'skipped')
                skipped.set('message', 'Test skipped')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        xml_path = self.output_dir / f"junit_report_{timestamp}.xml"
        
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        return str(xml_path)
    
    async def _generate_markdown_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        
        template = self.jinja_env.get_template('test_report.md')
        
        md_content = template.render(**data)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = self.output_dir / f"test_report_{timestamp}.md"
        
        async with aiofiles.open(md_path, 'w') as f:
            await f.write(md_content)
        
        return str(md_path)
    
    async def _generate_excel_report(self, suite: TestSuite, data: Dict[str, Any]) -> str:
        """Generate Excel report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = self.output_dir / f"test_report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Tests', 'Passed', 'Failed', 'Skipped', 'Success Rate', 'Duration', 'Coverage'],
                'Value': [suite.total_tests, suite.passed, suite.failed, suite.skipped, 
                         f"{suite.success_rate:.1f}%", f"{suite.total_duration:.2f}s", f"{suite.coverage_percentage:.1f}%"]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Test results sheet
            results_data = []
            for result in suite.results:
                results_data.append({
                    'Test Name': result.test_name,
                    'Module': result.test_module,
                    'Class': result.test_class,
                    'Status': result.status.value,
                    'Duration': result.duration,
                    'Start Time': result.start_time.isoformat(),
                    'End Time': result.end_time.isoformat(),
                    'Error Message': result.error_message or '',
                    'Markers': ','.join(result.markers) if result.markers else ''
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, sheet_name='Test Results', index=False)
            
            # Performance metrics sheet
            perf_data = {
                'Metric': list(data['performance_metrics'].keys()),
                'Value': list(data['performance_metrics'].values())
            }
            
            perf_df = pd.DataFrame(perf_data)
            perf_df.to_excel(writer, sheet_name='Performance', index=False)
            
            # Failed tests sheet
            failed_tests = [result for result in suite.results if result.status.value == 'failed']
            if failed_tests:
                failed_data = []
                for result in failed_tests:
                    failed_data.append({
                        'Test Name': result.test_name,
                        'Module': result.test_module,
                        'Duration': result.duration,
                        'Error Message': result.error_message or '',
                        'Stack Trace': result.stack_trace or ''
                    })
                
                failed_df = pd.DataFrame(failed_data)
                failed_df.to_excel(writer, sheet_name='Failed Tests', index=False)
        
        return str(excel_path)
    
    async def _create_archive(self, reports: Dict[str, str]) -> str:
        """Create compressed archive of all reports"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = self.output_dir / f"test_reports_{timestamp}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for format_name, report_path in reports.items():
                if format_name != 'archive' and Path(report_path).exists():
                    zf.write(report_path, Path(report_path).name)
            
            # Include chart files if they exist
            chart_dir = self.output_dir / "charts"
            if chart_dir.exists():
                for chart_file in chart_dir.glob("*.png"):
                    zf.write(chart_file, f"charts/{chart_file.name}")
        
        return str(archive_path)