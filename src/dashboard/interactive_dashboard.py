"""
Interactive Dashboard for GrandModel
===================================

Real-time interactive dashboard with performance monitoring, parameter adjustment,
multi-strategy comparison, risk monitoring, and drill-down analysis capabilities.

Features:
- Real-time performance monitoring
- Interactive parameter adjustment
- Multi-strategy comparison dashboard
- Risk monitoring with alerts
- Drill-down analysis capabilities
- Real-time data streaming
- Custom alerts and notifications
- Export capabilities
- Mobile-responsive design

Author: Agent 6 - Visualization and Reporting System
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import redis
import websocket
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
from sqlalchemy import create_engine, text
import warnings
from concurrent.futures import ThreadPoolExecutor
import queue

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization components
from ..visualization.advanced_visualization import AdvancedVisualization, ChartConfig
from ..reporting.comprehensive_reporting import ComprehensiveReporter


@dataclass
class DashboardConfig:
    """Configuration for dashboard"""
    port: int = 8050
    host: str = "0.0.0.0"
    debug: bool = False
    update_interval: int = 5  # seconds
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True
    theme: str = "dark"
    
    # Database configuration
    db_url: str = "sqlite:///dashboard.db"
    redis_url: str = "redis://localhost:6379"
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'max_drawdown': -0.10,  # 10% drawdown alert
                'volatility': 0.30,     # 30% volatility alert
                'sharpe_ratio': 0.5,    # Sharpe below 0.5 alert
                'var_95': -0.05         # 5% daily VaR alert
            }


class InteractiveDashboard:
    """
    Interactive dashboard for real-time monitoring and analysis
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize interactive dashboard
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        
        # Initialize components
        self.visualization = AdvancedVisualization()
        self.reporter = ComprehensiveReporter()
        
        # Initialize Flask app and Dash
        self.server = Flask(__name__)
        self.app = dash.Dash(__name__, 
                           server=self.server,
                           external_stylesheets=[dbc.themes.BOOTSTRAP if self.config.theme == 'light' else dbc.themes.CYBORG])
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.server, cors_allowed_origins="*")
        
        # Data storage
        self.data_queue = queue.Queue()
        self.current_data = {}
        self.strategies = {}
        self.alerts = []
        
        # Database connection
        self.db_engine = create_engine(self.config.db_url)
        self._init_database()
        
        # Redis connection for caching
        try:
            self.redis_client = redis.Redis.from_url(self.config.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Background tasks
        self.background_tasks = []
        self.dashboard_active = False
        
        # Setup dashboard
        self._setup_layout()
        self._setup_callbacks()
        self._setup_api_routes()
        
        logger.info("Interactive Dashboard initialized")
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with self.db_engine.connect() as conn:
                # Create tables
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        alert_message TEXT NOT NULL,
                        threshold_value REAL,
                        current_value REAL,
                        severity TEXT NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        
        # Create navigation bar
        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Overview", href="#overview")),
                dbc.NavItem(dbc.NavLink("Performance", href="#performance")),
                dbc.NavItem(dbc.NavLink("Risk", href="#risk")),
                dbc.NavItem(dbc.NavLink("Strategies", href="#strategies")),
                dbc.NavItem(dbc.NavLink("Alerts", href="#alerts")),
                dbc.NavItem(dbc.NavLink("Settings", href="#settings"))
            ],
            brand="GrandModel Interactive Dashboard",
            brand_href="#",
            color="primary",
            dark=True,
            className="mb-4"
        )
        
        # Create main content area
        main_content = dbc.Container([
            # Status indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Status", className="card-title"),
                            html.H2(id="system-status", className="text-success"),
                            html.P("Last updated: ", className="card-text"),
                            html.Span(id="last-update-time")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Strategies", className="card-title"),
                            html.H2(id="active-strategies-count", className="text-info"),
                            html.P("Currently monitored", className="card-text")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total P&L", className="card-title"),
                            html.H2(id="total-pnl", className="text-primary"),
                            html.P("Today's performance", className="card-text")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Alerts", className="card-title"),
                            html.H2(id="active-alerts-count", className="text-warning"),
                            html.P("Require attention", className="card-text")
                        ])
                    ], color="light", outline=True)
                ], width=3),
            ], className="mb-4"),
            
            # Main tabs
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview-tab"),
                dbc.Tab(label="Performance Analysis", tab_id="performance-tab"),
                dbc.Tab(label="Risk Monitoring", tab_id="risk-tab"),
                dbc.Tab(label="Strategy Comparison", tab_id="strategies-tab"),
                dbc.Tab(label="Real-time Monitoring", tab_id="realtime-tab"),
                dbc.Tab(label="Parameter Adjustment", tab_id="parameters-tab"),
                dbc.Tab(label="Alerts & Notifications", tab_id="alerts-tab"),
                dbc.Tab(label="Reports & Export", tab_id="reports-tab")
            ], id="main-tabs", active_tab="overview-tab"),
            
            # Tab content
            html.Div(id="tab-content", className="mt-4"),
            
            # Hidden components for data storage
            dcc.Store(id="dashboard-data-store"),
            dcc.Store(id="strategy-data-store"),
            dcc.Store(id="alert-data-store"),
            
            # Interval component for real-time updates
            dcc.Interval(
                id="update-interval",
                interval=self.config.update_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Modal for strategy configuration
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Strategy Configuration")),
                dbc.ModalBody([
                    html.Div(id="strategy-config-content")
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="save-strategy-config", color="primary"),
                    dbc.Button("Cancel", id="cancel-strategy-config", color="secondary")
                ])
            ], id="strategy-config-modal", is_open=False)
            
        ], fluid=True)
        
        # Set main layout
        self.app.layout = html.Div([
            navbar,
            main_content
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        # Main data update callback
        @self.app.callback(
            [Output('dashboard-data-store', 'data'),
             Output('strategy-data-store', 'data'),
             Output('alert-data-store', 'data')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_data_stores(n):
            """Update all data stores"""
            try:
                # Get current data
                dashboard_data = self._get_dashboard_data()
                strategy_data = self._get_strategy_data()
                alert_data = self._get_alert_data()
                
                return dashboard_data, strategy_data, alert_data
                
            except Exception as e:
                logger.error(f"Error updating data stores: {e}")
                return {}, {}, {}
        
        # Status indicators callback
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('active-strategies-count', 'children'),
             Output('total-pnl', 'children'),
             Output('active-alerts-count', 'children'),
             Output('last-update-time', 'children')],
            [Input('dashboard-data-store', 'data'),
             Input('strategy-data-store', 'data'),
             Input('alert-data-store', 'data')]
        )
        def update_status_indicators(dashboard_data, strategy_data, alert_data):
            """Update status indicator cards"""
            try:
                system_status = "Online" if dashboard_data else "Offline"
                active_strategies = len(strategy_data) if strategy_data else 0
                total_pnl = sum(s.get('pnl', 0) for s in strategy_data.values()) if strategy_data else 0
                active_alerts = len([a for a in alert_data if not a.get('acknowledged', False)]) if alert_data else 0
                last_update = datetime.now().strftime("%H:%M:%S")
                
                return (
                    system_status,
                    str(active_strategies),
                    f"${total_pnl:,.2f}",
                    str(active_alerts),
                    last_update
                )
                
            except Exception as e:
                logger.error(f"Error updating status indicators: {e}")
                return "Error", "0", "$0.00", "0", "Error"
        
        # Tab content callback
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('dashboard-data-store', 'data'),
             Input('strategy-data-store', 'data'),
             Input('alert-data-store', 'data')]
        )
        def update_tab_content(active_tab, dashboard_data, strategy_data, alert_data):
            """Update tab content based on selection"""
            try:
                if active_tab == "overview-tab":
                    return self._create_overview_tab(dashboard_data, strategy_data)
                elif active_tab == "performance-tab":
                    return self._create_performance_tab(strategy_data)
                elif active_tab == "risk-tab":
                    return self._create_risk_tab(strategy_data)
                elif active_tab == "strategies-tab":
                    return self._create_strategies_tab(strategy_data)
                elif active_tab == "realtime-tab":
                    return self._create_realtime_tab(dashboard_data)
                elif active_tab == "parameters-tab":
                    return self._create_parameters_tab(strategy_data)
                elif active_tab == "alerts-tab":
                    return self._create_alerts_tab(alert_data)
                elif active_tab == "reports-tab":
                    return self._create_reports_tab(strategy_data)
                else:
                    return html.Div("Select a tab to view content")
                    
            except Exception as e:
                logger.error(f"Error updating tab content: {e}")
                return html.Div(f"Error loading content: {e}")
    
    def _create_overview_tab(self, dashboard_data: Dict, strategy_data: Dict) -> html.Div:
        """Create overview tab content"""
        try:
            # Create overview charts
            overview_charts = []
            
            # Portfolio performance chart
            if strategy_data:
                portfolio_fig = self._create_portfolio_overview_chart(strategy_data)
                overview_charts.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Portfolio Performance"),
                            dbc.CardBody([
                                dcc.Graph(figure=portfolio_fig)
                            ])
                        ])
                    ], width=6)
                )
            
            # Risk metrics gauge
            risk_fig = self._create_risk_gauge_chart(strategy_data)
            overview_charts.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(figure=risk_fig)
                        ])
                    ])
                ], width=6)
            )
            
            # Strategy allocation pie chart
            allocation_fig = self._create_allocation_pie_chart(strategy_data)
            overview_charts.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Allocation"),
                        dbc.CardBody([
                            dcc.Graph(figure=allocation_fig)
                        ])
                    ])
                ], width=6)
            )
            
            # Recent performance table
            performance_table = self._create_performance_table(strategy_data)
            overview_charts.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Performance"),
                        dbc.CardBody([
                            performance_table
                        ])
                    ])
                ], width=6)
            )
            
            return html.Div([
                dbc.Row(overview_charts[:2], className="mb-4"),
                dbc.Row(overview_charts[2:], className="mb-4") if len(overview_charts) > 2 else html.Div()
            ])
            
        except Exception as e:
            logger.error(f"Error creating overview tab: {e}")
            return html.Div(f"Error creating overview: {e}")
    
    def _create_performance_tab(self, strategy_data: Dict) -> html.Div:
        """Create performance analysis tab content"""
        try:
            # Strategy selection dropdown
            strategy_options = [{'label': name, 'value': name} for name in strategy_data.keys()]
            
            performance_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Strategy:"),
                        dcc.Dropdown(
                            id="performance-strategy-dropdown",
                            options=strategy_options,
                            value=list(strategy_data.keys())[0] if strategy_data else None,
                            clearable=False
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Time Period:"),
                        dcc.Dropdown(
                            id="performance-period-dropdown",
                            options=[
                                {'label': '1 Day', 'value': 1},
                                {'label': '1 Week', 'value': 7},
                                {'label': '1 Month', 'value': 30},
                                {'label': '3 Months', 'value': 90},
                                {'label': '1 Year', 'value': 365}
                            ],
                            value=30,
                            clearable=False
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Benchmark:"),
                        dcc.Dropdown(
                            id="performance-benchmark-dropdown",
                            options=[
                                {'label': 'S&P 500', 'value': 'SPY'},
                                {'label': 'NASDAQ', 'value': 'QQQ'},
                                {'label': 'Custom', 'value': 'custom'}
                            ],
                            value='SPY',
                            clearable=True
                        )
                    ], width=4)
                ], className="mb-4"),
                
                # Performance charts
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Cumulative Returns"),
                            dbc.CardBody([
                                dcc.Graph(id="cumulative-returns-chart")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Drawdown Analysis"),
                            dbc.CardBody([
                                dcc.Graph(id="drawdown-chart")
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Performance Metrics"),
                            dbc.CardBody([
                                html.Div(id="performance-metrics-table")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Rolling Metrics"),
                            dbc.CardBody([
                                dcc.Graph(id="rolling-metrics-chart")
                            ])
                        ])
                    ], width=6)
                ])
            ]
            
            return html.Div(performance_content)
            
        except Exception as e:
            logger.error(f"Error creating performance tab: {e}")
            return html.Div(f"Error creating performance tab: {e}")
    
    def _create_risk_tab(self, strategy_data: Dict) -> html.Div:
        """Create risk monitoring tab content"""
        try:
            # Risk monitoring content
            risk_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Risk Overview"),
                            dbc.CardBody([
                                dcc.Graph(id="risk-overview-chart")
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Value at Risk (VaR)"),
                            dbc.CardBody([
                                dcc.Graph(id="var-chart")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Stress Testing"),
                            dbc.CardBody([
                                dcc.Graph(id="stress-test-chart")
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Risk Limits Monitoring"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="risk-limits-table",
                                    columns=[
                                        {"name": "Metric", "id": "metric"},
                                        {"name": "Current", "id": "current"},
                                        {"name": "Limit", "id": "limit"},
                                        {"name": "Utilization", "id": "utilization"},
                                        {"name": "Status", "id": "status"}
                                    ],
                                    style_cell={'textAlign': 'left'},
                                    style_data_conditional=[
                                        {
                                            'if': {'filter_query': '{status} = "Breach"'},
                                            'backgroundColor': '#ffcccc',
                                            'color': 'black',
                                        },
                                        {
                                            'if': {'filter_query': '{status} = "Warning"'},
                                            'backgroundColor': '#fff3cd',
                                            'color': 'black',
                                        }
                                    ]
                                )
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(risk_content)
            
        except Exception as e:
            logger.error(f"Error creating risk tab: {e}")
            return html.Div(f"Error creating risk tab: {e}")
    
    def _create_strategies_tab(self, strategy_data: Dict) -> html.Div:
        """Create strategy comparison tab content"""
        try:
            # Strategy comparison content
            strategies_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Strategy Performance Comparison"),
                            dbc.CardBody([
                                dcc.Graph(id="strategy-comparison-chart")
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Risk-Return Scatter"),
                            dbc.CardBody([
                                dcc.Graph(id="risk-return-scatter")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Correlation Matrix"),
                            dbc.CardBody([
                                dcc.Graph(id="correlation-matrix")
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Strategy Rankings"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="strategy-rankings-table",
                                    columns=[
                                        {"name": "Strategy", "id": "strategy"},
                                        {"name": "Total Return", "id": "total_return", "type": "numeric", "format": {"specifier": ".2%"}},
                                        {"name": "Sharpe Ratio", "id": "sharpe_ratio", "type": "numeric", "format": {"specifier": ".3f"}},
                                        {"name": "Max Drawdown", "id": "max_drawdown", "type": "numeric", "format": {"specifier": ".2%"}},
                                        {"name": "Volatility", "id": "volatility", "type": "numeric", "format": {"specifier": ".2%"}},
                                        {"name": "Rank", "id": "rank"}
                                    ],
                                    sort_action="native",
                                    style_cell={'textAlign': 'left'}
                                )
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(strategies_content)
            
        except Exception as e:
            logger.error(f"Error creating strategies tab: {e}")
            return html.Div(f"Error creating strategies tab: {e}")
    
    def _create_realtime_tab(self, dashboard_data: Dict) -> html.Div:
        """Create real-time monitoring tab content"""
        try:
            # Real-time monitoring content
            realtime_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Live P&L"),
                            dbc.CardBody([
                                dcc.Graph(id="live-pnl-chart")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Live Positions"),
                            dbc.CardBody([
                                dcc.Graph(id="live-positions-chart")
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Market Data Stream"),
                            dbc.CardBody([
                                dcc.Graph(id="market-data-stream")
                            ])
                        ])
                    ], width=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Health"),
                            dbc.CardBody([
                                html.Div(id="system-health-indicators")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Recent Trades"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="recent-trades-table",
                                    columns=[
                                        {"name": "Time", "id": "time"},
                                        {"name": "Strategy", "id": "strategy"},
                                        {"name": "Symbol", "id": "symbol"},
                                        {"name": "Side", "id": "side"},
                                        {"name": "Size", "id": "size"},
                                        {"name": "Price", "id": "price"},
                                        {"name": "P&L", "id": "pnl"}
                                    ],
                                    style_cell={'textAlign': 'left'},
                                    page_size=10
                                )
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(realtime_content)
            
        except Exception as e:
            logger.error(f"Error creating real-time tab: {e}")
            return html.Div(f"Error creating real-time tab: {e}")
    
    def _create_parameters_tab(self, strategy_data: Dict) -> html.Div:
        """Create parameter adjustment tab content"""
        try:
            # Parameter adjustment content
            parameters_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Strategy Parameters"),
                            dbc.CardBody([
                                html.Div(id="strategy-parameters-form")
                            ])
                        ])
                    ], width=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Parameter History"),
                            dbc.CardBody([
                                html.Div(id="parameter-history")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Parameter Impact Analysis"),
                            dbc.CardBody([
                                dcc.Graph(id="parameter-impact-chart")
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(parameters_content)
            
        except Exception as e:
            logger.error(f"Error creating parameters tab: {e}")
            return html.Div(f"Error creating parameters tab: {e}")
    
    def _create_alerts_tab(self, alert_data: List) -> html.Div:
        """Create alerts and notifications tab content"""
        try:
            # Alerts content
            alerts_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Alert Configuration"),
                            dbc.CardBody([
                                html.Div(id="alert-configuration")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Alert Statistics"),
                            dbc.CardBody([
                                dcc.Graph(id="alert-statistics-chart")
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Active Alerts"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="active-alerts-table",
                                    columns=[
                                        {"name": "Time", "id": "time"},
                                        {"name": "Strategy", "id": "strategy"},
                                        {"name": "Type", "id": "type"},
                                        {"name": "Message", "id": "message"},
                                        {"name": "Severity", "id": "severity"},
                                        {"name": "Status", "id": "status"},
                                        {"name": "Actions", "id": "actions"}
                                    ],
                                    style_cell={'textAlign': 'left'},
                                    page_size=10
                                )
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(alerts_content)
            
        except Exception as e:
            logger.error(f"Error creating alerts tab: {e}")
            return html.Div(f"Error creating alerts tab: {e}")
    
    def _create_reports_tab(self, strategy_data: Dict) -> html.Div:
        """Create reports and export tab content"""
        try:
            # Reports content
            reports_content = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Report Generation"),
                            dbc.CardBody([
                                dbc.Form([
                                    dbc.Row([
                                        dbc.Label("Report Type", width=3),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id="report-type-dropdown",
                                                options=[
                                                    {"label": "Executive Summary", "value": "executive"},
                                                    {"label": "Performance Analysis", "value": "performance"},
                                                    {"label": "Risk Assessment", "value": "risk"},
                                                    {"label": "Strategy Comparison", "value": "comparison"}
                                                ],
                                                value="executive"
                                            )
                                        ], width=9)
                                    ], className="mb-3"),
                                    
                                    dbc.Row([
                                        dbc.Label("Time Period", width=3),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id="report-period-dropdown",
                                                options=[
                                                    {"label": "Last 7 Days", "value": 7},
                                                    {"label": "Last 30 Days", "value": 30},
                                                    {"label": "Last 90 Days", "value": 90},
                                                    {"label": "Last Year", "value": 365}
                                                ],
                                                value=30
                                            )
                                        ], width=9)
                                    ], className="mb-3"),
                                    
                                    dbc.Row([
                                        dbc.Label("Output Format", width=3),
                                        dbc.Col([
                                            dcc.Dropdown(
                                                id="report-format-dropdown",
                                                options=[
                                                    {"label": "PDF", "value": "pdf"},
                                                    {"label": "HTML", "value": "html"},
                                                    {"label": "JSON", "value": "json"},
                                                    {"label": "Excel", "value": "excel"}
                                                ],
                                                value="pdf"
                                            )
                                        ], width=9)
                                    ], className="mb-3"),
                                    
                                    dbc.Button("Generate Report", id="generate-report-btn", color="primary", size="lg")
                                ])
                            ])
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Report History"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id="report-history-table",
                                    columns=[
                                        {"name": "Date", "id": "date"},
                                        {"name": "Type", "id": "type"},
                                        {"name": "Format", "id": "format"},
                                        {"name": "Size", "id": "size"},
                                        {"name": "Actions", "id": "actions"}
                                    ],
                                    style_cell={'textAlign': 'left'},
                                    page_size=10
                                )
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Export Options"),
                            dbc.CardBody([
                                html.Div(id="export-options")
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            return html.Div(reports_content)
            
        except Exception as e:
            logger.error(f"Error creating reports tab: {e}")
            return html.Div(f"Error creating reports tab: {e}")
    
    def _setup_api_routes(self):
        """Setup API routes for dashboard"""
        
        @self.server.route('/api/data', methods=['GET'])
        def get_data():
            """Get current dashboard data"""
            try:
                return jsonify(self.current_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.server.route('/api/strategies', methods=['GET'])
        def get_strategies():
            """Get strategy data"""
            try:
                return jsonify(self.strategies)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.server.route('/api/alerts', methods=['GET'])
        def get_alerts():
            """Get alert data"""
            try:
                return jsonify(self.alerts)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.server.route('/api/update_parameters', methods=['POST'])
        def update_parameters():
            """Update strategy parameters"""
            try:
                data = request.get_json()
                strategy_name = data.get('strategy_name')
                parameters = data.get('parameters')
                
                # Update parameters logic would go here
                # This is a placeholder
                
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        try:
            # This would typically fetch from database or real-time data source
            # For now, return mock data
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'online',
                'total_strategies': len(self.strategies),
                'total_pnl': sum(s.get('pnl', 0) for s in self.strategies.values()),
                'active_alerts': len(self.alerts)
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _get_strategy_data(self) -> Dict[str, Any]:
        """Get strategy data"""
        try:
            # This would typically fetch from database
            # For now, return mock data
            return self.strategies
        except Exception as e:
            logger.error(f"Error getting strategy data: {e}")
            return {}
    
    def _get_alert_data(self) -> List[Dict[str, Any]]:
        """Get alert data"""
        try:
            # This would typically fetch from database
            # For now, return mock data
            return self.alerts
        except Exception as e:
            logger.error(f"Error getting alert data: {e}")
            return []
    
    def _create_portfolio_overview_chart(self, strategy_data: Dict) -> go.Figure:
        """Create portfolio overview chart"""
        try:
            fig = go.Figure()
            
            # Add mock data for now
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating portfolio overview chart: {e}")
            return go.Figure()
    
    def _create_risk_gauge_chart(self, strategy_data: Dict) -> go.Figure:
        """Create risk gauge chart"""
        try:
            # Mock risk score
            risk_score = 65
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk gauge chart: {e}")
            return go.Figure()
    
    def _create_allocation_pie_chart(self, strategy_data: Dict) -> go.Figure:
        """Create allocation pie chart"""
        try:
            # Mock allocation data
            labels = ['Strategy A', 'Strategy B', 'Strategy C']
            values = [40, 35, 25]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(
                title="Strategy Allocation",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating allocation pie chart: {e}")
            return go.Figure()
    
    def _create_performance_table(self, strategy_data: Dict) -> dash_table.DataTable:
        """Create performance table"""
        try:
            # Mock performance data
            data = [
                {"strategy": "Strategy A", "return": "12.5%", "sharpe": "1.45", "drawdown": "-5.2%"},
                {"strategy": "Strategy B", "return": "8.7%", "sharpe": "1.12", "drawdown": "-3.1%"},
                {"strategy": "Strategy C", "return": "15.3%", "sharpe": "1.78", "drawdown": "-7.8%"}
            ]
            
            return dash_table.DataTable(
                data=data,
                columns=[
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Return", "id": "return"},
                    {"name": "Sharpe", "id": "sharpe"},
                    {"name": "Max DD", "id": "drawdown"}
                ],
                style_cell={'textAlign': 'left'}
            )
            
        except Exception as e:
            logger.error(f"Error creating performance table: {e}")
            return dash_table.DataTable(data=[], columns=[])
    
    def start_dashboard(self):
        """Start the dashboard server"""
        try:
            self.dashboard_active = True
            
            # Start background tasks
            self._start_background_tasks()
            
            logger.info(f"Starting dashboard on {self.config.host}:{self.config.port}")
            
            # Run the dashboard
            self.app.run_server(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def stop_dashboard(self):
        """Stop the dashboard server"""
        try:
            self.dashboard_active = False
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            logger.info("Dashboard stopped")
            
        except Exception as e:
            logger.error(f"Error stopping dashboard: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for data updates"""
        try:
            # Start data update task
            async def update_data():
                while self.dashboard_active:
                    try:
                        # Update current data
                        self.current_data = self._get_dashboard_data()
                        
                        # Emit real-time updates via SocketIO
                        if self.config.enable_real_time:
                            self.socketio.emit('data_update', self.current_data)
                        
                        await asyncio.sleep(self.config.update_interval)
                        
                    except Exception as e:
                        logger.error(f"Error in data update task: {e}")
                        await asyncio.sleep(1)
            
            # Start alert monitoring task
            async def monitor_alerts():
                while self.dashboard_active:
                    try:
                        # Check for new alerts
                        self._check_alerts()
                        
                        await asyncio.sleep(10)  # Check every 10 seconds
                        
                    except Exception as e:
                        logger.error(f"Error in alert monitoring task: {e}")
                        await asyncio.sleep(1)
            
            # Create tasks
            update_task = asyncio.create_task(update_data())
            alert_task = asyncio.create_task(monitor_alerts())
            
            self.background_tasks = [update_task, alert_task]
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def _check_alerts(self):
        """Check for alerts based on thresholds"""
        try:
            # This would check current metrics against thresholds
            # and generate alerts if breached
            pass
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")


# Global instance
interactive_dashboard = InteractiveDashboard()