"""
Notebook Integration System for GrandModel
==========================================

Seamless integration with existing notebook and backtesting systems,
providing embedded visualization, reporting, and analysis capabilities.

Features:
- Jupyter notebook integration
- Backtesting system integration
- Embedded visualization widgets
- Real-time data streaming to notebooks
- Interactive parameter adjustment
- Automated report generation
- Export capabilities
- Data pipeline integration

Author: Agent 6 - Visualization and Reporting System
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import threading
from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from jupyter_dash import JupyterDash
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing components
from ..visualization.advanced_visualization import AdvancedVisualization, ChartConfig
from ..reporting.comprehensive_reporting import ComprehensiveReporter
from ..dashboard.interactive_dashboard import InteractiveDashboard
from ..integration.export_integration import ExportIntegration


@dataclass
class NotebookConfig:
    """Configuration for notebook integration"""
    jupyter_theme: str = "light"
    auto_refresh: bool = True
    refresh_interval: int = 5  # seconds
    enable_widgets: bool = True
    enable_real_time: bool = True
    max_chart_width: int = 1000
    max_chart_height: int = 600
    chart_template: str = "plotly_white"
    
    # Integration settings
    vectorbt_integration: bool = True
    backtesting_integration: bool = True
    data_pipeline_integration: bool = True
    
    # Export settings
    auto_export: bool = False
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['html', 'png']


class NotebookIntegration:
    """
    Comprehensive notebook integration system
    """
    
    def __init__(self, config: Optional[NotebookConfig] = None):
        """
        Initialize notebook integration
        
        Args:
            config: Notebook configuration
        """
        self.config = config or NotebookConfig()
        
        # Initialize components
        self.visualization = AdvancedVisualization()
        self.reporter = ComprehensiveReporter()
        self.exporter = ExportIntegration()
        
        # Notebook state
        self.current_data = {}
        self.widgets = {}
        self.figures = {}
        self.reports = {}
        
        # Background tasks
        self.update_tasks = []
        self.is_running = False
        
        logger.info("Notebook Integration System initialized")
    
    def create_performance_widget(self, 
                                strategy_data: Dict[str, Any],
                                widget_id: str = "performance_widget") -> widgets.VBox:
        """
        Create interactive performance analysis widget
        
        Args:
            strategy_data: Strategy performance data
            widget_id: Widget identifier
            
        Returns:
            Interactive performance widget
        """
        try:
            # Create widget components
            title = widgets.HTML(value="<h2>Performance Analysis</h2>")
            
            # Strategy selector
            strategy_names = list(strategy_data.keys()) if isinstance(strategy_data, dict) else ["Strategy"]
            strategy_selector = widgets.Dropdown(
                options=strategy_names,
                value=strategy_names[0] if strategy_names else "Strategy",
                description="Strategy:",
                disabled=False
            )
            
            # Time period selector
            period_selector = widgets.Dropdown(
                options=[
                    ("1 Day", 1),
                    ("1 Week", 7),
                    ("1 Month", 30),
                    ("3 Months", 90),
                    ("1 Year", 365)
                ],
                value=30,
                description="Period:",
                disabled=False
            )
            
            # Chart type selector
            chart_selector = widgets.Dropdown(
                options=[
                    ("Performance Overview", "performance"),
                    ("Risk Analysis", "risk"),
                    ("Drawdown Analysis", "drawdown"),
                    ("Return Distribution", "returns")
                ],
                value="performance",
                description="Chart Type:",
                disabled=False
            )
            
            # Output area
            output = widgets.Output()
            
            # Update function
            def update_performance_widget(change=None):
                with output:
                    output.clear_output()
                    
                    try:
                        # Get selected values
                        selected_strategy = strategy_selector.value
                        selected_period = period_selector.value
                        selected_chart = chart_selector.value
                        
                        # Generate chart based on selection
                        if selected_chart == "performance":
                            fig = self._create_performance_chart(strategy_data, selected_strategy, selected_period)
                        elif selected_chart == "risk":
                            fig = self._create_risk_chart(strategy_data, selected_strategy, selected_period)
                        elif selected_chart == "drawdown":
                            fig = self._create_drawdown_chart(strategy_data, selected_strategy, selected_period)
                        elif selected_chart == "returns":
                            fig = self._create_returns_chart(strategy_data, selected_strategy, selected_period)
                        else:
                            fig = go.Figure()
                        
                        # Display chart
                        fig.show()
                        
                        # Store figure
                        self.figures[f"{widget_id}_{selected_chart}"] = fig
                        
                    except Exception as e:
                        print(f"Error updating widget: {e}")
            
            # Bind events
            strategy_selector.observe(update_performance_widget, names='value')
            period_selector.observe(update_performance_widget, names='value')
            chart_selector.observe(update_performance_widget, names='value')
            
            # Initial update
            update_performance_widget()
            
            # Create widget
            widget = widgets.VBox([
                title,
                widgets.HBox([strategy_selector, period_selector, chart_selector]),
                output
            ])
            
            # Store widget
            self.widgets[widget_id] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating performance widget: {e}")
            return widgets.VBox([widgets.HTML(value=f"<p>Error creating widget: {e}</p>")])
    
    def create_risk_monitor_widget(self, 
                                 strategy_data: Dict[str, Any],
                                 widget_id: str = "risk_monitor") -> widgets.VBox:
        """
        Create interactive risk monitoring widget
        
        Args:
            strategy_data: Strategy data
            widget_id: Widget identifier
            
        Returns:
            Interactive risk monitoring widget
        """
        try:
            # Create widget components
            title = widgets.HTML(value="<h2>Risk Monitor</h2>")
            
            # Risk metric selector
            risk_metrics = [
                ("Value at Risk (VaR)", "var"),
                ("Expected Shortfall", "es"),
                ("Maximum Drawdown", "mdd"),
                ("Volatility", "vol"),
                ("Sharpe Ratio", "sharpe")
            ]
            
            metric_selector = widgets.Dropdown(
                options=risk_metrics,
                value="var",
                description="Risk Metric:",
                disabled=False
            )
            
            # Confidence level slider (for VaR/ES)
            confidence_slider = widgets.FloatSlider(
                value=0.95,
                min=0.90,
                max=0.99,
                step=0.01,
                description="Confidence:",
                disabled=False,
                continuous_update=False
            )
            
            # Time window selector
            window_selector = widgets.Dropdown(
                options=[
                    ("30 Days", 30),
                    ("60 Days", 60),
                    ("90 Days", 90),
                    ("180 Days", 180),
                    ("365 Days", 365)
                ],
                value=90,
                description="Time Window:",
                disabled=False
            )
            
            # Output area
            output = widgets.Output()
            
            # Update function
            def update_risk_monitor(change=None):
                with output:
                    output.clear_output()
                    
                    try:
                        # Get selected values
                        selected_metric = metric_selector.value
                        confidence_level = confidence_slider.value
                        time_window = window_selector.value
                        
                        # Generate risk chart
                        fig = self._create_risk_monitor_chart(
                            strategy_data, selected_metric, confidence_level, time_window
                        )
                        
                        # Display chart
                        fig.show()
                        
                        # Calculate and display risk statistics
                        risk_stats = self._calculate_risk_statistics(
                            strategy_data, selected_metric, confidence_level, time_window
                        )
                        
                        # Display statistics
                        stats_html = self._format_risk_statistics(risk_stats)
                        display(HTML(stats_html))
                        
                        # Store figure
                        self.figures[f"{widget_id}_{selected_metric}"] = fig
                        
                    except Exception as e:
                        print(f"Error updating risk monitor: {e}")
            
            # Bind events
            metric_selector.observe(update_risk_monitor, names='value')
            confidence_slider.observe(update_risk_monitor, names='value')
            window_selector.observe(update_risk_monitor, names='value')
            
            # Initial update
            update_risk_monitor()
            
            # Create widget
            widget = widgets.VBox([
                title,
                widgets.HBox([metric_selector, confidence_slider, window_selector]),
                output
            ])
            
            # Store widget
            self.widgets[widget_id] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating risk monitor widget: {e}")
            return widgets.VBox([widgets.HTML(value=f"<p>Error creating widget: {e}</p>")])
    
    def create_strategy_comparison_widget(self, 
                                        strategies: Dict[str, Dict[str, Any]],
                                        widget_id: str = "strategy_comparison") -> widgets.VBox:
        """
        Create interactive strategy comparison widget
        
        Args:
            strategies: Dictionary of strategies
            widget_id: Widget identifier
            
        Returns:
            Interactive strategy comparison widget
        """
        try:
            # Create widget components
            title = widgets.HTML(value="<h2>Strategy Comparison</h2>")
            
            # Strategy selection
            strategy_names = list(strategies.keys())
            strategy_selector = widgets.SelectMultiple(
                options=strategy_names,
                value=strategy_names[:3] if len(strategy_names) >= 3 else strategy_names,
                description="Strategies:",
                disabled=False
            )
            
            # Comparison type selector
            comparison_types = [
                ("Performance", "performance"),
                ("Risk Metrics", "risk"),
                ("Risk-Return", "risk_return"),
                ("Correlation", "correlation")
            ]
            
            comparison_selector = widgets.Dropdown(
                options=comparison_types,
                value="performance",
                description="Comparison:",
                disabled=False
            )
            
            # Benchmark selector
            benchmarks = [
                ("None", "none"),
                ("S&P 500", "spy"),
                ("NASDAQ", "qqq"),
                ("Custom", "custom")
            ]
            
            benchmark_selector = widgets.Dropdown(
                options=benchmarks,
                value="none",
                description="Benchmark:",
                disabled=False
            )
            
            # Output area
            output = widgets.Output()
            
            # Update function
            def update_comparison(change=None):
                with output:
                    output.clear_output()
                    
                    try:
                        # Get selected values
                        selected_strategies = list(strategy_selector.value)
                        comparison_type = comparison_selector.value
                        benchmark = benchmark_selector.value
                        
                        # Filter strategies
                        filtered_strategies = {k: v for k, v in strategies.items() if k in selected_strategies}
                        
                        # Generate comparison chart
                        if comparison_type == "performance":
                            fig = self._create_performance_comparison_chart(filtered_strategies, benchmark)
                        elif comparison_type == "risk":
                            fig = self._create_risk_comparison_chart(filtered_strategies)
                        elif comparison_type == "risk_return":
                            fig = self._create_risk_return_scatter(filtered_strategies)
                        elif comparison_type == "correlation":
                            fig = self._create_correlation_heatmap(filtered_strategies)
                        else:
                            fig = go.Figure()
                        
                        # Display chart
                        fig.show()
                        
                        # Generate comparison table
                        comparison_table = self._create_comparison_table(filtered_strategies)
                        display(HTML(comparison_table))
                        
                        # Store figure
                        self.figures[f"{widget_id}_{comparison_type}"] = fig
                        
                    except Exception as e:
                        print(f"Error updating comparison: {e}")
            
            # Bind events
            strategy_selector.observe(update_comparison, names='value')
            comparison_selector.observe(update_comparison, names='value')
            benchmark_selector.observe(update_comparison, names='value')
            
            # Initial update
            update_comparison()
            
            # Create widget
            widget = widgets.VBox([
                title,
                widgets.HBox([strategy_selector, comparison_selector, benchmark_selector]),
                output
            ])
            
            # Store widget
            self.widgets[widget_id] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating strategy comparison widget: {e}")
            return widgets.VBox([widgets.HTML(value=f"<p>Error creating widget: {e}</p>")])
    
    def create_embedded_dashboard(self, 
                                data: Dict[str, Any],
                                dashboard_id: str = "embedded_dashboard") -> JupyterDash:
        """
        Create embedded dashboard in notebook
        
        Args:
            data: Dashboard data
            dashboard_id: Dashboard identifier
            
        Returns:
            Embedded dashboard app
        """
        try:
            # Create Jupyter Dash app
            app = JupyterDash(__name__)
            
            # Define layout
            app.layout = html.Div([
                html.H1("GrandModel Dashboard", className="text-center mb-4"),
                
                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Strategy:"),
                        dcc.Dropdown(
                            id="strategy-dropdown",
                            options=[{"label": k, "value": k} for k in data.keys()],
                            value=list(data.keys())[0] if data else None
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Select Period:"),
                        dcc.Dropdown(
                            id="period-dropdown",
                            options=[
                                {"label": "1 Month", "value": 30},
                                {"label": "3 Months", "value": 90},
                                {"label": "1 Year", "value": 365}
                            ],
                            value=90
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Refresh:"),
                        dbc.Button("Refresh Data", id="refresh-btn", color="primary")
                    ], width=4)
                ], className="mb-4"),
                
                # Charts
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="performance-chart")
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id="risk-chart")
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="drawdown-chart")
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id="returns-chart")
                    ], width=6)
                ])
            ])
            
            # Callbacks
            @app.callback(
                [Output("performance-chart", "figure"),
                 Output("risk-chart", "figure"),
                 Output("drawdown-chart", "figure"),
                 Output("returns-chart", "figure")],
                [Input("strategy-dropdown", "value"),
                 Input("period-dropdown", "value"),
                 Input("refresh-btn", "n_clicks")]
            )
            def update_charts(strategy, period, n_clicks):
                try:
                    # Generate charts
                    perf_fig = self._create_performance_chart(data, strategy, period)
                    risk_fig = self._create_risk_chart(data, strategy, period)
                    dd_fig = self._create_drawdown_chart(data, strategy, period)
                    ret_fig = self._create_returns_chart(data, strategy, period)
                    
                    return perf_fig, risk_fig, dd_fig, ret_fig
                    
                except Exception as e:
                    logger.error(f"Error updating dashboard charts: {e}")
                    return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            
            # Store dashboard
            self.widgets[dashboard_id] = app
            
            return app
            
        except Exception as e:
            logger.error(f"Error creating embedded dashboard: {e}")
            return None
    
    def integrate_with_vectorbt(self, 
                              vectorbt_portfolio: Any,
                              widget_id: str = "vectorbt_integration") -> widgets.VBox:
        """
        Integrate with VectorBT portfolio
        
        Args:
            vectorbt_portfolio: VectorBT portfolio object
            widget_id: Widget identifier
            
        Returns:
            VectorBT integration widget
        """
        try:
            # Create widget components
            title = widgets.HTML(value="<h2>VectorBT Integration</h2>")
            
            # Portfolio statistics
            stats = vectorbt_portfolio.stats()
            stats_html = self._format_vectorbt_stats(stats)
            stats_display = widgets.HTML(value=stats_html)
            
            # Chart selector
            chart_types = [
                ("Portfolio Value", "value"),
                ("Returns", "returns"),
                ("Drawdown", "drawdown"),
                ("Trade Analysis", "trades")
            ]
            
            chart_selector = widgets.Dropdown(
                options=chart_types,
                value="value",
                description="Chart Type:",
                disabled=False
            )
            
            # Output area
            output = widgets.Output()
            
            # Update function
            def update_vectorbt_chart(change=None):
                with output:
                    output.clear_output()
                    
                    try:
                        chart_type = chart_selector.value
                        
                        if chart_type == "value":
                            fig = self._create_vectorbt_value_chart(vectorbt_portfolio)
                        elif chart_type == "returns":
                            fig = self._create_vectorbt_returns_chart(vectorbt_portfolio)
                        elif chart_type == "drawdown":
                            fig = self._create_vectorbt_drawdown_chart(vectorbt_portfolio)
                        elif chart_type == "trades":
                            fig = self._create_vectorbt_trades_chart(vectorbt_portfolio)
                        else:
                            fig = go.Figure()
                        
                        fig.show()
                        
                        # Store figure
                        self.figures[f"{widget_id}_{chart_type}"] = fig
                        
                    except Exception as e:
                        print(f"Error updating VectorBT chart: {e}")
            
            # Bind events
            chart_selector.observe(update_vectorbt_chart, names='value')
            
            # Initial update
            update_vectorbt_chart()
            
            # Create widget
            widget = widgets.VBox([
                title,
                stats_display,
                chart_selector,
                output
            ])
            
            # Store widget
            self.widgets[widget_id] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"Error integrating with VectorBT: {e}")
            return widgets.VBox([widgets.HTML(value=f"<p>Error integrating with VectorBT: {e}</p>")])
    
    def create_report_generator_widget(self, 
                                     strategy_data: Dict[str, Any],
                                     widget_id: str = "report_generator") -> widgets.VBox:
        """
        Create report generator widget
        
        Args:
            strategy_data: Strategy data
            widget_id: Widget identifier
            
        Returns:
            Report generator widget
        """
        try:
            # Create widget components
            title = widgets.HTML(value="<h2>Report Generator</h2>")
            
            # Report type selector
            report_types = [
                ("Executive Summary", "executive"),
                ("Performance Analysis", "performance"),
                ("Risk Assessment", "risk"),
                ("Strategy Comparison", "comparison")
            ]
            
            report_selector = widgets.Dropdown(
                options=report_types,
                value="executive",
                description="Report Type:",
                disabled=False
            )
            
            # Format selector
            format_options = [
                ("HTML", "html"),
                ("PDF", "pdf"),
                ("PowerPoint", "pptx"),
                ("JSON", "json")
            ]
            
            format_selector = widgets.Dropdown(
                options=format_options,
                value="html",
                description="Format:",
                disabled=False
            )
            
            # Generate button
            generate_btn = widgets.Button(
                description="Generate Report",
                button_style="success",
                icon="file"
            )
            
            # Output area
            output = widgets.Output()
            
            # Generate function
            def generate_report(btn):
                with output:
                    output.clear_output()
                    
                    try:
                        report_type = report_selector.value
                        format_type = format_selector.value
                        
                        print(f"Generating {report_type} report in {format_type} format...")
                        
                        # Generate report
                        if report_type == "executive":
                            report = self.reporter.generate_executive_summary(strategy_data)
                        elif report_type == "performance":
                            report = self.reporter.generate_detailed_performance_report(strategy_data)
                        elif report_type == "risk":
                            report = self.reporter.generate_risk_assessment_report(strategy_data)
                        elif report_type == "comparison":
                            report = self.reporter.generate_strategy_comparison_report(strategy_data)
                        else:
                            report = {}
                        
                        # Export report
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{report_type}_report_{timestamp}"
                        
                        if format_type == "html":
                            file_path = self.exporter.export_to_html(report, filename)
                        elif format_type == "pdf":
                            file_path = self.exporter.export_to_pdf(report, filename)
                        elif format_type == "json":
                            file_path = self.exporter.export_to_json(report, filename)
                        else:
                            file_path = ""
                        
                        if file_path:
                            print(f"Report generated successfully: {file_path}")
                            
                            # Display HTML report in notebook
                            if format_type == "html":
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                display(HTML(html_content))
                        else:
                            print("Error generating report")
                        
                        # Store report
                        self.reports[f"{widget_id}_{report_type}"] = report
                        
                    except Exception as e:
                        print(f"Error generating report: {e}")
            
            # Bind events
            generate_btn.on_click(generate_report)
            
            # Create widget
            widget = widgets.VBox([
                title,
                widgets.HBox([report_selector, format_selector]),
                generate_btn,
                output
            ])
            
            # Store widget
            self.widgets[widget_id] = widget
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating report generator widget: {e}")
            return widgets.VBox([widgets.HTML(value=f"<p>Error creating widget: {e}</p>")])
    
    def auto_refresh_data(self, data_source: callable, interval: int = 30):
        """
        Start auto-refresh of data
        
        Args:
            data_source: Function that returns updated data
            interval: Refresh interval in seconds
        """
        try:
            async def refresh_loop():
                while self.is_running:
                    try:
                        # Get updated data
                        updated_data = data_source()
                        
                        # Update current data
                        self.current_data.update(updated_data)
                        
                        # Trigger widget updates
                        self._trigger_widget_updates()
                        
                        # Wait for next refresh
                        await asyncio.sleep(interval)
                        
                    except Exception as e:
                        logger.error(f"Error in auto-refresh: {e}")
                        await asyncio.sleep(5)
            
            # Start refresh task
            self.is_running = True
            task = asyncio.create_task(refresh_loop())
            self.update_tasks.append(task)
            
            logger.info(f"Auto-refresh started with {interval}s interval")
            
        except Exception as e:
            logger.error(f"Error starting auto-refresh: {e}")
    
    def stop_auto_refresh(self):
        """Stop auto-refresh"""
        try:
            self.is_running = False
            
            # Cancel all update tasks
            for task in self.update_tasks:
                task.cancel()
            
            self.update_tasks.clear()
            
            logger.info("Auto-refresh stopped")
            
        except Exception as e:
            logger.error(f"Error stopping auto-refresh: {e}")
    
    def export_notebook_report(self, 
                             notebook_path: str,
                             report_data: Dict[str, Any],
                             include_figures: bool = True) -> str:
        """
        Export notebook with embedded report
        
        Args:
            notebook_path: Output notebook path
            report_data: Report data
            include_figures: Whether to include figures
            
        Returns:
            Path to generated notebook
        """
        try:
            # Create new notebook
            nb = new_notebook()
            
            # Add title cell
            title_cell = new_markdown_cell(f"# {report_data.get('report_metadata', {}).get('strategy_name', 'Strategy')} Analysis Report")
            nb.cells.append(title_cell)
            
            # Add imports cell
            imports_code = """
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
"""
            imports_cell = new_code_cell(imports_code)
            nb.cells.append(imports_cell)
            
            # Add data cell
            data_code = f"report_data = {json.dumps(report_data, indent=2, default=str)}"
            data_cell = new_code_cell(data_code)
            nb.cells.append(data_cell)
            
            # Add analysis sections
            sections = [
                ("Executive Summary", "executive_summary"),
                ("Key Metrics", "key_metrics"),
                ("Performance Analysis", "performance_analysis"),
                ("Risk Assessment", "risk_assessment"),
                ("Conclusions", "conclusions")
            ]
            
            for section_title, section_key in sections:
                if section_key in report_data:
                    # Section header
                    header_cell = new_markdown_cell(f"## {section_title}")
                    nb.cells.append(header_cell)
                    
                    # Section content
                    if section_key == "key_metrics":
                        metrics_code = f"""
# Display key metrics
metrics = report_data.get('{section_key}', {{}})
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        if 'return' in key.lower() or 'ratio' in key.lower():
            print(f"{{key.replace('_', ' ').title()}}: {{value:.2%}}")
        else:
            print(f"{{key.replace('_', ' ').title()}}: {{value:.4f}}")
    else:
        print(f"{{key.replace('_', ' ').title()}}: {{value}}")
"""
                        metrics_cell = new_code_cell(metrics_code)
                        nb.cells.append(metrics_cell)
                    
                    elif section_key == "performance_analysis":
                        perf_code = f"""
# Performance analysis
perf_data = report_data.get('{section_key}', {{}})
for analysis_type, analysis_data in perf_data.items():
    if isinstance(analysis_data, dict):
        print(f"\\n**{{analysis_type.replace('_', ' ').title()}}:**")
        for key, value in analysis_data.items():
            if isinstance(value, (int, float)):
                print(f"  {{key.replace('_', ' ').title()}}: {{value:.4f}}")
            else:
                print(f"  {{key.replace('_', ' ').title()}}: {{value}}")
"""
                        perf_cell = new_code_cell(perf_code)
                        nb.cells.append(perf_cell)
            
            # Add figures if requested
            if include_figures and self.figures:
                figures_cell = new_markdown_cell("## Charts and Visualizations")
                nb.cells.append(figures_cell)
                
                for fig_name, fig in self.figures.items():
                    fig_code = f"""
# {fig_name.replace('_', ' ').title()}
# Note: This would display the actual figure in a real notebook
print("Figure: {fig_name}")
"""
                    fig_cell = new_code_cell(fig_code)
                    nb.cells.append(fig_cell)
            
            # Save notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            logger.info(f"Notebook report exported to {notebook_path}")
            return notebook_path
            
        except Exception as e:
            logger.error(f"Error exporting notebook report: {e}")
            return ""
    
    def get_widget_state(self) -> Dict[str, Any]:
        """Get current widget state"""
        try:
            state = {
                'widgets': list(self.widgets.keys()),
                'figures': list(self.figures.keys()),
                'reports': list(self.reports.keys()),
                'current_data': self.current_data,
                'is_running': self.is_running
            }
            return state
            
        except Exception as e:
            logger.error(f"Error getting widget state: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop auto-refresh
            self.stop_auto_refresh()
            
            # Clear widgets
            self.widgets.clear()
            self.figures.clear()
            self.reports.clear()
            
            logger.info("Notebook integration cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
    # Helper methods for chart creation
    def _create_performance_chart(self, data: Dict[str, Any], strategy: str, period: int) -> go.Figure:
        """Create performance chart"""
        try:
            # Mock performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=period), end=datetime.now(), freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            cumulative_returns = np.cumprod(1 + returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name=f'{strategy} Performance',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f'{strategy} Performance ({period} days)',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return go.Figure()
    
    def _create_risk_chart(self, data: Dict[str, Any], strategy: str, period: int) -> go.Figure:
        """Create risk chart"""
        try:
            # Mock risk data
            dates = pd.date_range(start=datetime.now() - timedelta(days=period), end=datetime.now(), freq='D')
            volatility = np.random.normal(0.15, 0.05, len(dates))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=volatility,
                mode='lines',
                name=f'{strategy} Volatility',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'{strategy} Risk Metrics ({period} days)',
                xaxis_title='Date',
                yaxis_title='Volatility',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk chart: {e}")
            return go.Figure()
    
    def _create_drawdown_chart(self, data: Dict[str, Any], strategy: str, period: int) -> go.Figure:
        """Create drawdown chart"""
        try:
            # Mock drawdown data
            dates = pd.date_range(start=datetime.now() - timedelta(days=period), end=datetime.now(), freq='D')
            drawdown = np.random.normal(-0.02, 0.01, len(dates))
            drawdown = np.minimum(drawdown, 0)  # Ensure non-positive
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name=f'{strategy} Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)'
            ))
            
            fig.update_layout(
                title=f'{strategy} Drawdown ({period} days)',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return go.Figure()
    
    def _create_returns_chart(self, data: Dict[str, Any], strategy: str, period: int) -> go.Figure:
        """Create returns distribution chart"""
        try:
            # Mock returns data
            returns = np.random.normal(0.001, 0.02, period)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=30,
                name=f'{strategy} Returns',
                marker_color='blue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'{strategy} Returns Distribution ({period} days)',
                xaxis_title='Daily Returns',
                yaxis_title='Frequency',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating returns chart: {e}")
            return go.Figure()
    
    def _trigger_widget_updates(self):
        """Trigger updates for all widgets"""
        try:
            # This would trigger updates for all active widgets
            # Implementation depends on specific widget types
            pass
            
        except Exception as e:
            logger.error(f"Error triggering widget updates: {e}")
    
    # Additional helper methods would continue here...
    
    def _format_vectorbt_stats(self, stats: Any) -> str:
        """Format VectorBT statistics for display"""
        try:
            stats_html = "<div class='vectorbt-stats'>"
            stats_html += "<h3>Portfolio Statistics</h3>"
            stats_html += "<table style='width: 100%; border-collapse: collapse;'>"
            
            # Convert stats to dict if needed
            if hasattr(stats, 'to_dict'):
                stats_dict = stats.to_dict()
            else:
                stats_dict = dict(stats)
            
            for key, value in stats_dict.items():
                if isinstance(value, (int, float)):
                    if 'return' in key.lower() or 'ratio' in key.lower():
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                stats_html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{key}</td>"
                stats_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{formatted_value}</td></tr>"
            
            stats_html += "</table></div>"
            return stats_html
            
        except Exception as e:
            logger.error(f"Error formatting VectorBT stats: {e}")
            return "<p>Error formatting statistics</p>"


# Global instance
notebook_integration = NotebookIntegration()