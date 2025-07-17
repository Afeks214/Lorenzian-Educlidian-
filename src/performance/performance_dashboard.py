"""
Automated Performance Reporting Dashboard

This module provides a comprehensive web-based dashboard for performance monitoring,
reporting, and visualization with real-time updates and automated report generation.

Features:
- Real-time performance metrics visualization
- Interactive charts and graphs
- Automated report generation
- Performance trend analysis
- SLA monitoring dashboard
- Alert management interface
- Historical performance data
- Export capabilities

Author: Performance Validation Agent
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import structlog
from dataclasses import asdict
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from flask import Flask, send_file
import threading
import time
from jinja2 import Template
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Import our performance components
from .comprehensive_validation_framework import performance_validator
from .enhanced_regression_detection import enhanced_regression_detector
from .continuous_benchmarking_pipeline import continuous_benchmarking

logger = structlog.get_logger()

class PerformanceDashboard:
    """
    Web-based performance dashboard with real-time monitoring and reporting
    """

    def __init__(self, port: int = 8050, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.server = self.app.server
        
        # Dashboard state
        self.dashboard_active = False
        self.update_interval = 5  # seconds
        self.last_update = datetime.now()
        
        # Initialize dashboard
        self._setup_layout()
        self._setup_callbacks()
        
        # Background tasks
        self.background_thread = None
        
        logger.info("Performance dashboard initialized", port=port)

    def _setup_layout(self):
        """Setup dashboard layout"""
        
        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H1("GrandModel Performance Dashboard", className="text-light")),
                    dbc.Col(html.Div(id="last-update", className="text-light"), width="auto")
                ], align="center", className="g-0"),
            ]),
            color="primary",
            dark=True,
            className="mb-4"
        )
        
        # Metrics summary cards
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("System Health", className="card-title"),
                        html.H2(id="system-health-score", className="text-success"),
                        html.P("Overall system performance", className="card-text")
                    ])
                ], color="light", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Alerts", className="card-title"),
                        html.H2(id="active-alerts-count", className="text-warning"),
                        html.P("Performance violations", className="card-text")
                    ])
                ], color="light", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Tests Passing", className="card-title"),
                        html.H2(id="tests-passing-percent", className="text-info"),
                        html.P("SLA compliance rate", className="card-text")
                    ])
                ], color="light", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Regressions", className="card-title"),
                        html.H2(id="regressions-count", className="text-danger"),
                        html.P("Recent performance drops", className="card-text")
                    ])
                ], color="light", outline=True)
            ], width=3),
        ], className="mb-4")
        
        # Main content tabs
        tabs = dbc.Tabs([
            dbc.Tab(label="Real-time Metrics", tab_id="realtime", active_tab_style={"textTransform": "uppercase"}),
            dbc.Tab(label="Performance Trends", tab_id="trends", active_tab_style={"textTransform": "uppercase"}),
            dbc.Tab(label="SLA Monitoring", tab_id="sla", active_tab_style={"textTransform": "uppercase"}),
            dbc.Tab(label="Regression Analysis", tab_id="regression", active_tab_style={"textTransform": "uppercase"}),
            dbc.Tab(label="Load Testing", tab_id="load", active_tab_style={"textTransform": "uppercase"}),
            dbc.Tab(label="Reports", tab_id="reports", active_tab_style={"textTransform": "uppercase"})
        ], id="main-tabs", active_tab="realtime")
        
        # Tab content
        tab_content = html.Div(id="tab-content", className="mt-3")
        
        # Main layout
        self.app.layout = dbc.Container([
            header,
            metrics_cards,
            tabs,
            tab_content,
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            ),
            dcc.Store(id='dashboard-data-store')
        ], fluid=True)

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('dashboard-data-store', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard_data(n):
            """Update dashboard data store"""
            return self._get_dashboard_data()
        
        @self.app.callback(
            [Output('system-health-score', 'children'),
             Output('active-alerts-count', 'children'),
             Output('tests-passing-percent', 'children'),
             Output('regressions-count', 'children'),
             Output('last-update', 'children')],
            Input('dashboard-data-store', 'data')
        )
        def update_metrics_cards(data):
            """Update metrics summary cards"""
            if not data:
                return "N/A", "N/A", "N/A", "N/A", "Never"
            
            health_score = f"{data.get('system_health_score', 0):.1f}%"
            alerts_count = str(data.get('active_alerts', 0))
            passing_percent = f"{data.get('sla_compliance_rate', 0):.1f}%"
            regressions_count = str(data.get('recent_regressions', 0))
            last_update = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
            
            return health_score, alerts_count, passing_percent, regressions_count, last_update
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('dashboard-data-store', 'data')]
        )
        def update_tab_content(active_tab, data):
            """Update tab content based on selection"""
            if active_tab == "realtime":
                return self._create_realtime_tab(data)
            elif active_tab == "trends":
                return self._create_trends_tab(data)
            elif active_tab == "sla":
                return self._create_sla_tab(data)
            elif active_tab == "regression":
                return self._create_regression_tab(data)
            elif active_tab == "load":
                return self._create_load_tab(data)
            elif active_tab == "reports":
                return self._create_reports_tab(data)
            else:
                return html.Div("Select a tab")

    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get performance report
            perf_report = performance_validator.generate_performance_report(hours=24)
            
            # Get regression summary
            regression_summary = enhanced_regression_detector.get_regression_summary(hours=24)
            
            # Get pipeline status
            pipeline_status = continuous_benchmarking.get_pipeline_status()
            
            # Calculate derived metrics
            system_health_score = perf_report.get('system_health_score', 0)
            active_alerts = perf_report.get('summary', {}).get('total_violations', 0)
            
            # Calculate SLA compliance rate
            total_tests = perf_report.get('summary', {}).get('total_tests', 1)
            passing_tests = 0
            for test_name, test_data in perf_report.get('test_summaries', {}).items():
                for metric_type, metric_data in test_data.items():
                    if metric_data.get('target_met_rate', 0) > 0.8:  # 80% threshold
                        passing_tests += 1
            
            sla_compliance_rate = (passing_tests / max(total_tests, 1)) * 100
            
            recent_regressions = regression_summary.get('total_regressions', 0)
            
            return {
                'system_health_score': system_health_score,
                'active_alerts': active_alerts,
                'sla_compliance_rate': sla_compliance_rate,
                'recent_regressions': recent_regressions,
                'performance_report': perf_report,
                'regression_summary': regression_summary,
                'pipeline_status': pipeline_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting dashboard data", error=str(e))
            return {}

    def _create_realtime_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create real-time metrics tab"""
        if not data:
            return html.Div("No data available")
        
        perf_report = data.get('performance_report', {})
        test_summaries = perf_report.get('test_summaries', {})
        
        # Create performance charts
        charts = []
        
        for test_name, test_data in test_summaries.items():
            if 'latency' in test_data:
                # Latency chart
                fig = go.Figure()
                
                latency_data = test_data['latency']
                fig.add_trace(go.Scatter(
                    x=['P50', 'P95', 'P99', 'Max'],
                    y=[
                        latency_data.get('avg', 0),
                        latency_data.get('p95', 0),
                        latency_data.get('p99', 0),
                        latency_data.get('max', 0)
                    ],
                    mode='lines+markers',
                    name=f'{test_name} Latency',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title=f'{test_name} Latency Distribution',
                    xaxis_title='Percentile',
                    yaxis_title='Latency (ms)',
                    height=300
                )
                
                charts.append(
                    dbc.Col([
                        dcc.Graph(figure=fig)
                    ], width=6)
                )
        
        if not charts:
            return html.Div("No performance data available")
        
        return dbc.Container([
            dbc.Row(charts),
            html.Hr(),
            html.H4("Current System Status"),
            self._create_system_status_table(data)
        ])

    def _create_trends_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create performance trends tab"""
        if not data:
            return html.Div("No data available")
        
        # Create trend analysis charts
        return dbc.Container([
            html.H4("Performance Trends (24 hours)"),
            html.P("Historical performance data and trend analysis"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Latency Trends"),
                    dcc.Graph(id="latency-trends-chart")
                ], width=6),
                dbc.Col([
                    html.H5("Throughput Trends"),
                    dcc.Graph(id="throughput-trends-chart")
                ], width=6)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Memory Usage Trends"),
                    dcc.Graph(id="memory-trends-chart")
                ], width=6),
                dbc.Col([
                    html.H5("CPU Usage Trends"),
                    dcc.Graph(id="cpu-trends-chart")
                ], width=6)
            ])
        ])

    def _create_sla_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create SLA monitoring tab"""
        if not data:
            return html.Div("No data available")
        
        perf_report = data.get('performance_report', {})
        performance_targets = perf_report.get('performance_targets', {})
        
        # Create SLA status table
        sla_data = []
        for target_name, target_info in performance_targets.items():
            sla_data.append({
                'Target': target_name,
                'Max Latency (ms)': target_info.get('max_latency_ms', 'N/A'),
                'Min Throughput (ops/sec)': target_info.get('min_throughput_ops_per_sec', 'N/A'),
                'Max Memory (MB)': target_info.get('max_memory_mb', 'N/A'),
                'Status': '✅ Compliant' if target_name in perf_report.get('test_summaries', {}) else '❌ No Data'
            })
        
        sla_table = dash_table.DataTable(
            data=sla_data,
            columns=[{"name": i, "id": i} for i in sla_data[0].keys() if sla_data],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Status} contains "Compliant"'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Status} contains "No Data"'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
        
        return dbc.Container([
            html.H4("SLA Monitoring"),
            html.P("Service Level Agreement compliance status"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("SLA Targets"),
                    sla_table
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Recent Violations"),
                    html.Div(id="violations-list")
                ], width=12)
            ])
        ])

    def _create_regression_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create regression analysis tab"""
        if not data:
            return html.Div("No data available")
        
        regression_summary = data.get('regression_summary', {})
        
        # Create regression severity chart
        severity_data = regression_summary.get('severity_breakdown', {})
        
        if severity_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(severity_data.keys()),
                    y=list(severity_data.values()),
                    marker_color=['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severity_data.keys()]
                )
            ])
            
            fig.update_layout(
                title="Regression Severity Distribution",
                xaxis_title="Severity Level",
                yaxis_title="Count",
                height=400
            )
            
            severity_chart = dcc.Graph(figure=fig)
        else:
            severity_chart = html.Div("No regression data available")
        
        # Top regressed tests
        top_regressions = regression_summary.get('top_regressed_tests', [])
        
        regression_table = dash_table.DataTable(
            data=top_regressions,
            columns=[
                {"name": "Test", "id": "test_name"},
                {"name": "Metric", "id": "metric_name"},
                {"name": "Regression Count", "id": "regression_count"},
                {"name": "Avg Confidence", "id": "avg_confidence", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            style_cell={'textAlign': 'left'},
            sort_action="native"
        )
        
        return dbc.Container([
            html.H4("Regression Analysis"),
            html.P("Performance regression detection and analysis"),
            
            dbc.Row([
                dbc.Col([
                    severity_chart
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Top Regressed Tests"),
                    regression_table
                ], width=12)
            ])
        ])

    def _create_load_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create load testing tab"""
        return dbc.Container([
            html.H4("Load Testing"),
            html.P("Load testing results and performance under stress"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Load Test Configuration"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Test Duration (seconds)", width=3),
                            dbc.Col([
                                dcc.Input(id="load-duration", type="number", value=60, min=10, max=3600)
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Concurrent Users", width=3),
                            dbc.Col([
                                dcc.Input(id="load-users", type="number", value=10, min=1, max=1000)
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Requests/Second", width=3),
                            dbc.Col([
                                dcc.Input(id="load-rps", type="number", value=100, min=1, max=10000)
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Button("Start Load Test", id="start-load-test", color="primary")
                    ])
                ], width=6),
                dbc.Col([
                    html.H5("Load Test Results"),
                    html.Div(id="load-test-results")
                ], width=6)
            ])
        ])

    def _create_reports_tab(self, data: Dict[str, Any]) -> html.Div:
        """Create reports tab"""
        return dbc.Container([
            html.H4("Automated Reports"),
            html.P("Generate and download performance reports"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Report Generation"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Report Type", width=3),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="report-type",
                                    options=[
                                        {"label": "Performance Summary", "value": "performance"},
                                        {"label": "Regression Analysis", "value": "regression"},
                                        {"label": "SLA Compliance", "value": "sla"},
                                        {"label": "Load Testing", "value": "load"}
                                    ],
                                    value="performance"
                                )
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Time Period", width=3),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="report-period",
                                    options=[
                                        {"label": "Last 24 Hours", "value": 24},
                                        {"label": "Last 7 Days", "value": 168},
                                        {"label": "Last 30 Days", "value": 720}
                                    ],
                                    value=24
                                )
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Format", width=3),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="report-format",
                                    options=[
                                        {"label": "PDF", "value": "pdf"},
                                        {"label": "HTML", "value": "html"},
                                        {"label": "JSON", "value": "json"}
                                    ],
                                    value="pdf"
                                )
                            ], width=9)
                        ], className="mb-3"),
                        dbc.Button("Generate Report", id="generate-report", color="primary")
                    ])
                ], width=6),
                dbc.Col([
                    html.H5("Report History"),
                    html.Div(id="report-history")
                ], width=6)
            ])
        ])

    def _create_system_status_table(self, data: Dict[str, Any]) -> dash_table.DataTable:
        """Create system status table"""
        perf_report = data.get('performance_report', {})
        test_summaries = perf_report.get('test_summaries', {})
        
        status_data = []
        for test_name, test_data in test_summaries.items():
            for metric_type, metric_info in test_data.items():
                if isinstance(metric_info, dict):
                    status_data.append({
                        'Test': test_name,
                        'Metric': metric_type,
                        'Current': f"{metric_info.get('avg', 0):.2f}",
                        'Target Met': f"{metric_info.get('target_met_rate', 0):.1%}",
                        'Status': '✅ Good' if metric_info.get('target_met_rate', 0) > 0.8 else '⚠️ Warning' if metric_info.get('target_met_rate', 0) > 0.5 else '❌ Critical'
                    })
        
        return dash_table.DataTable(
            data=status_data,
            columns=[{"name": i, "id": i} for i in status_data[0].keys() if status_data],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Status} contains "Good"'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Status} contains "Warning"'},
                    'backgroundColor': '#fff3cd',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Status} contains "Critical"'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ],
            page_size=10
        )

    def start_dashboard(self):
        """Start the dashboard server"""
        if self.dashboard_active:
            logger.warning("Dashboard is already running")
            return
        
        self.dashboard_active = True
        
        # Start background tasks
        self.background_thread = threading.Thread(target=self._background_tasks)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        logger.info("Starting performance dashboard", port=self.port)
        
        # Run the dashboard
        self.app.run_server(host='0.0.0.0', port=self.port, debug=self.debug)

    def stop_dashboard(self):
        """Stop the dashboard server"""
        self.dashboard_active = False
        logger.info("Performance dashboard stopped")

    def _background_tasks(self):
        """Background tasks for dashboard"""
        while self.dashboard_active:
            try:
                # Generate automatic reports
                self._generate_scheduled_reports()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error("Error in background tasks", error=str(e))
                time.sleep(60)

    def _generate_scheduled_reports(self):
        """Generate scheduled reports"""
        try:
            # Generate daily performance summary
            current_hour = datetime.now().hour
            if current_hour == 0:  # Midnight
                report_data = self._get_dashboard_data()
                self._generate_performance_report(report_data, "daily")
            
            # Generate weekly summary
            current_day = datetime.now().weekday()
            if current_day == 6 and current_hour == 0:  # Sunday midnight
                report_data = self._get_dashboard_data()
                self._generate_performance_report(report_data, "weekly")
                
        except Exception as e:
            logger.error("Error generating scheduled reports", error=str(e))

    def _generate_performance_report(self, data: Dict[str, Any], report_type: str):
        """Generate performance report"""
        try:
            # Create report content
            report_content = self._create_report_content(data, report_type)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{report_type}_{timestamp}.html"
            filepath = Path(f"reports/{filename}")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            logger.info("Performance report generated",
                       report_type=report_type,
                       filename=filename)
            
        except Exception as e:
            logger.error("Error generating performance report", error=str(e))

    def _create_report_content(self, data: Dict[str, Any], report_type: str) -> str:
        """Create HTML report content"""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {{ report_type.title() }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #007bff; color: white; padding: 20px; text-align: center; }
                .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }
                .good { border-left-color: #28a745; }
                .warning { border-left-color: #ffc107; }
                .critical { border-left-color: #dc3545; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GrandModel Performance Report</h1>
                <p>{{ report_type.title() }} Report - {{ timestamp }}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>System Health Score:</strong> {{ data.get('system_health_score', 0) }}%
            </div>
            <div class="metric">
                <strong>SLA Compliance Rate:</strong> {{ data.get('sla_compliance_rate', 0) }}%
            </div>
            <div class="metric">
                <strong>Active Alerts:</strong> {{ data.get('active_alerts', 0) }}
            </div>
            <div class="metric">
                <strong>Recent Regressions:</strong> {{ data.get('recent_regressions', 0) }}
            </div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Metric</th>
                    <th>Current Value</th>
                    <th>Target Met Rate</th>
                    <th>Status</th>
                </tr>
                {% for test_name, test_data in data.get('performance_report', {}).get('test_summaries', {}).items() %}
                    {% for metric_type, metric_info in test_data.items() %}
                        {% if metric_info is mapping %}
                        <tr>
                            <td>{{ test_name }}</td>
                            <td>{{ metric_type }}</td>
                            <td>{{ "%.2f"|format(metric_info.get('avg', 0)) }}</td>
                            <td>{{ "%.1f%%"|format(metric_info.get('target_met_rate', 0) * 100) }}</td>
                            <td>{{ "Good" if metric_info.get('target_met_rate', 0) > 0.8 else "Warning" if metric_info.get('target_met_rate', 0) > 0.5 else "Critical" }}</td>
                        </tr>
                        {% endif %}
                    {% endfor %}
                {% endfor %}
            </table>
            
            <h2>Regression Analysis</h2>
            {% set regression_data = data.get('regression_summary', {}) %}
            <p>Total Regressions: {{ regression_data.get('total_regressions', 0) }}</p>
            <p>Severity Breakdown:</p>
            <ul>
                {% for severity, count in regression_data.get('severity_breakdown', {}).items() %}
                <li>{{ severity }}: {{ count }}</li>
                {% endfor %}
            </ul>
            
            <div style="margin-top: 50px; text-align: center; color: #666;">
                <p>Generated automatically by GrandModel Performance Dashboard</p>
                <p>{{ timestamp }}</p>
            </div>
        </body>
        </html>
        """)
        
        return template.render(
            data=data,
            report_type=report_type,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            # Clean up data older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # This would typically clean up database records
            # Implementation depends on the specific database structure
            
            logger.debug("Cleaned up old performance data")
            
        except Exception as e:
            logger.error("Error cleaning up old data", error=str(e))


# Global instance
performance_dashboard = PerformanceDashboard()