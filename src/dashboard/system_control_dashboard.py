"""
System Control Dashboard for GrandModel
========================================

Web-based dashboard with visual switch interface for system control including:
- Large ON/OFF toggle switch
- Real-time status indicators
- Component health dashboard
- Recent activity log
- Performance metrics display
- Emergency stop button
- Authentication interface
- Mobile-responsive design

This dashboard provides a modern, secure, and user-friendly interface for
system control operations with real-time updates and comprehensive monitoring.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import jwt
from flask import Flask, session, request as flask_request
from flask_session import Session
import redis
import json
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    api_base_url: str = "http://localhost:8001"
    status_api_url: str = "http://localhost:8002"
    refresh_interval: int = 5  # seconds
    websocket_url: str = "ws://localhost:8002/ws/status"
    theme: str = "dark"
    debug: bool = False
    secret_key: str = "your-secret-key-here"
    redis_url: str = "redis://localhost:6379"
    session_timeout: int = 3600  # seconds

class SystemState(str, Enum):
    """System state enumeration"""
    OFF = "OFF"
    ON = "ON"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MAINTENANCE = "MAINTENANCE"

class SystemControlDashboard:
    """
    System control dashboard with visual interface
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard"""
        self.config = config or DashboardConfig()
        
        # Initialize Flask server
        self.server = Flask(__name__)
        self.server.config['SECRET_KEY'] = self.config.secret_key
        self.server.config['SESSION_TYPE'] = 'redis'
        self.server.config['SESSION_REDIS'] = redis.from_url(self.config.redis_url)
        self.server.config['SESSION_PERMANENT'] = False
        self.server.config['SESSION_USE_SIGNER'] = True
        self.server.config['SESSION_KEY_PREFIX'] = 'grandmodel:'
        
        # Initialize session
        Session(self.server)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[
                dbc.themes.CYBORG if self.config.theme == "dark" else dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
                "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
            ],
            suppress_callback_exceptions=True
        )
        
        # Set app title
        self.app.title = "GrandModel System Control"
        
        # Initialize data stores
        self.current_user = None
        self.system_status = None
        self.component_health = {}
        self.performance_metrics = {}
        self.alerts = []
        self.activity_logs = []
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        self._setup_auth_routes()
        
        logger.info("System Control Dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # URL routing
            dcc.Location(id='url', refresh=False),
            
            # Main content area
            html.Div(id='page-content'),
            
            # Hidden components for data storage
            dcc.Store(id='user-session', storage_type='session'),
            dcc.Store(id='system-data-store'),
            dcc.Store(id='component-data-store'),
            dcc.Store(id='alerts-data-store'),
            dcc.Store(id='performance-data-store'),
            
            # Interval components for real-time updates
            dcc.Interval(
                id='status-update-interval',
                interval=self.config.refresh_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Confirmation modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Confirm Action")),
                dbc.ModalBody([
                    html.P(id="confirmation-message"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Reason:"),
                        dbc.Input(id="action-reason", placeholder="Enter reason for this action")
                    ], className="mb-3"),
                    dbc.Checklist(
                        id="force-action-check",
                        options=[{"label": "Force action (override safety checks)", "value": "force"}],
                        value=[]
                    )
                ]),
                dbc.ModalFooter([
                    dbc.Button("Confirm", id="confirm-action-btn", color="primary"),
                    dbc.Button("Cancel", id="cancel-action-btn", color="secondary")
                ])
            ], id="confirmation-modal", is_open=False),
            
            # Alert toast
            dbc.Toast(
                id="alert-toast",
                header="System Alert",
                is_open=False,
                dismissable=True,
                duration=5000,
                icon="danger",
                style={"position": "fixed", "top": 20, "right": 20, "width": 350, "z-index": 9999}
            ),
            
            # Success toast
            dbc.Toast(
                id="success-toast",
                header="Success",
                is_open=False,
                dismissable=True,
                duration=3000,
                icon="success",
                style={"position": "fixed", "top": 20, "right": 20, "width": 350, "z-index": 9999}
            )
        ])
    
    def _create_login_page(self):
        """Create login page"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2("GrandModel System Control", className="text-center mb-4"),
                            html.P("Please log in to access the system control dashboard.", 
                                  className="text-center text-muted mb-4"),
                            
                            dbc.Form([
                                dbc.InputGroup([
                                    dbc.InputGroupText(html.I(className="fas fa-user")),
                                    dbc.Input(
                                        id="username-input",
                                        placeholder="Username",
                                        type="text",
                                        required=True
                                    )
                                ], className="mb-3"),
                                
                                dbc.InputGroup([
                                    dbc.InputGroupText(html.I(className="fas fa-lock")),
                                    dbc.Input(
                                        id="password-input",
                                        placeholder="Password",
                                        type="password",
                                        required=True
                                    )
                                ], className="mb-3"),
                                
                                dbc.Button(
                                    "Login",
                                    id="login-btn",
                                    color="primary",
                                    className="w-100",
                                    size="lg"
                                ),
                                
                                html.Div(id="login-error", className="text-danger mt-2")
                            ])
                        ])
                    ], className="shadow")
                ], width=6, lg=4)
            ], justify="center", className="min-vh-100 align-items-center")
        ], fluid=True)
    
    def _create_main_dashboard(self):
        """Create main dashboard page"""
        return html.Div([
            # Header
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem([
                        dbc.Button(
                            html.I(className="fas fa-sign-out-alt"),
                            id="logout-btn",
                            color="outline-light",
                            size="sm"
                        )
                    ])
                ],
                brand="GrandModel System Control",
                brand_href="#",
                color="primary",
                dark=True,
                className="mb-4"
            ),
            
            dbc.Container([
                # System Status Row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4([
                                    html.I(className="fas fa-power-off me-2"),
                                    "System Control"
                                ])
                            ]),
                            dbc.CardBody([
                                # Main power switch
                                html.Div([
                                    html.Div([
                                        dbc.Switch(
                                            id="main-power-switch",
                                            value=False,
                                            className="custom-switch",
                                            style={"transform": "scale(2)", "margin": "20px"}
                                        )
                                    ], className="text-center mb-3"),
                                    
                                    html.H5(id="system-status-text", className="text-center"),
                                    
                                    html.Div([
                                        dbc.Button(
                                            [html.I(className="fas fa-exclamation-triangle me-2"), "EMERGENCY STOP"],
                                            id="emergency-stop-btn",
                                            color="danger",
                                            size="lg",
                                            className="w-100 mb-2"
                                        ),
                                        dbc.Button(
                                            [html.I(className="fas fa-tools me-2"), "Maintenance Mode"],
                                            id="maintenance-btn",
                                            color="warning",
                                            size="sm",
                                            className="w-100"
                                        )
                                    ])
                                ])
                            ])
                        ])
                    ], width=12, lg=4),
                    
                    dbc.Col([
                        # System overview cards
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("System Uptime"),
                                        html.H4(id="system-uptime", className="text-success"),
                                        html.Small("Last restart: ", className="text-muted"),
                                        html.Small(id="last-restart-time", className="text-muted")
                                    ])
                                ], className="text-center")
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Health Score"),
                                        html.H4(id="health-score", className="text-info"),
                                        html.Small("Overall system health")
                                    ])
                                ], className="text-center")
                            ], width=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Active Alerts"),
                                        html.H4(id="active-alerts-count", className="text-warning"),
                                        html.Small("Require attention")
                                    ])
                                ], className="text-center")
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Performance"),
                                        html.H4(id="performance-score", className="text-primary"),
                                        html.Small("Current performance")
                                    ])
                                ], className="text-center")
                            ], width=6)
                        ])
                    ], width=12, lg=8)
                ], className="mb-4"),
                
                # Component Health Row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-heartbeat me-2"),
                                    "Component Health"
                                ])
                            ]),
                            dbc.CardBody([
                                html.Div(id="component-health-grid")
                            ])
                        ])
                    ], width=12, lg=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Performance Metrics"
                                ])
                            ]),
                            dbc.CardBody([
                                dcc.Graph(id="performance-gauge-chart", style={"height": "300px"})
                            ])
                        ])
                    ], width=12, lg=4)
                ], className="mb-4"),
                
                # Alerts and Activity Row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-bell me-2"),
                                    "Active Alerts"
                                ])
                            ]),
                            dbc.CardBody([
                                html.Div(id="alerts-list")
                            ])
                        ])
                    ], width=12, lg=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-history me-2"),
                                    "Recent Activity"
                                ])
                            ]),
                            dbc.CardBody([
                                html.Div(id="activity-log")
                            ])
                        ])
                    ], width=12, lg=6)
                ], className="mb-4"),
                
                # System Metrics Chart
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-chart-area me-2"),
                                    "System Metrics"
                                ])
                            ]),
                            dbc.CardBody([
                                dcc.Graph(id="system-metrics-chart", style={"height": "400px"})
                            ])
                        ])
                    ], width=12)
                ])
            ], fluid=True)
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        # Page routing callback
        @self.app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname'),
             Input('user-session', 'data')]
        )
        def display_page(pathname, user_data):
            if user_data and user_data.get('authenticated'):
                return self._create_main_dashboard()
            else:
                return self._create_login_page()
        
        # Login callback
        @self.app.callback(
            [Output('user-session', 'data'),
             Output('login-error', 'children')],
            [Input('login-btn', 'n_clicks')],
            [State('username-input', 'value'),
             State('password-input', 'value')]
        )
        def handle_login(n_clicks, username, password):
            if not n_clicks:
                return None, ""
            
            if not username or not password:
                return None, "Please enter both username and password"
            
            # Simple authentication (in production, use proper auth)
            if username == "admin" and password == "admin123":
                user_data = {
                    'authenticated': True,
                    'username': username,
                    'login_time': datetime.now().isoformat(),
                    'permissions': ['system_control', 'read', 'write']
                }
                return user_data, ""
            else:
                return None, "Invalid username or password"
        
        # Logout callback
        @self.app.callback(
            Output('user-session', 'data', allow_duplicate=True),
            [Input('logout-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_logout(n_clicks):
            if n_clicks:
                return None
            return dash.no_update
        
        # Data fetching callback
        @self.app.callback(
            [Output('system-data-store', 'data'),
             Output('component-data-store', 'data'),
             Output('alerts-data-store', 'data'),
             Output('performance-data-store', 'data')],
            [Input('status-update-interval', 'n_intervals')],
            [State('user-session', 'data')]
        )
        def update_data_stores(n_intervals, user_data):
            if not user_data or not user_data.get('authenticated'):
                return {}, {}, [], {}
            
            try:
                # Fetch system status
                system_data = self._fetch_system_status()
                
                # Fetch component health
                component_data = self._fetch_component_health()
                
                # Fetch alerts
                alerts_data = self._fetch_alerts()
                
                # Fetch performance metrics
                performance_data = self._fetch_performance_metrics()
                
                return system_data, component_data, alerts_data, performance_data
                
            except Exception as e:
                logger.error(f"Error updating data stores: {e}")
                return {}, {}, [], {}
        
        # System status display callback
        @self.app.callback(
            [Output('main-power-switch', 'value'),
             Output('system-status-text', 'children'),
             Output('system-status-text', 'className'),
             Output('system-uptime', 'children'),
             Output('last-restart-time', 'children'),
             Output('health-score', 'children'),
             Output('active-alerts-count', 'children'),
             Output('performance-score', 'children')],
            [Input('system-data-store', 'data'),
             Input('alerts-data-store', 'data')]
        )
        def update_system_status(system_data, alerts_data):
            if not system_data:
                return False, "Unknown", "text-muted", "N/A", "N/A", "N/A", "0", "N/A"
            
            # Power switch state
            switch_value = system_data.get('status') == 'ON'
            
            # Status text and color
            status = system_data.get('status', 'Unknown')
            if status == 'ON':
                status_text = "System Online"
                status_class = "text-success"
            elif status == 'OFF':
                status_text = "System Offline"
                status_class = "text-danger"
            elif status == 'STARTING':
                status_text = "System Starting..."
                status_class = "text-warning"
            elif status == 'STOPPING':
                status_text = "System Stopping..."
                status_class = "text-warning"
            elif status == 'EMERGENCY_STOP':
                status_text = "EMERGENCY STOP"
                status_class = "text-danger"
            else:
                status_text = f"Status: {status}"
                status_class = "text-muted"
            
            # Uptime
            uptime_seconds = system_data.get('uptime_seconds', 0)
            uptime_str = self._format_uptime(uptime_seconds)
            
            # Last restart
            last_restart = system_data.get('last_operation_time', 'Unknown')
            
            # Health score
            health_score = system_data.get('performance_metrics', {}).get('overall_health', 0)
            health_str = f"{health_score:.1f}%"
            
            # Active alerts
            active_alerts = len([a for a in alerts_data if a.get('status') == 'active'])
            
            # Performance score
            performance_score = system_data.get('performance_metrics', {}).get('performance_score', 0)
            performance_str = f"{performance_score:.1f}%"
            
            return (
                switch_value,
                status_text,
                status_class,
                uptime_str,
                last_restart,
                health_str,
                str(active_alerts),
                performance_str
            )
        
        # Component health display callback
        @self.app.callback(
            Output('component-health-grid', 'children'),
            [Input('component-data-store', 'data')]
        )
        def update_component_health(component_data):
            if not component_data:
                return html.P("No component data available")
            
            components = []
            for component in component_data:
                health_score = component.get('health_score', 0)
                status = component.get('status', 'unknown')
                
                # Determine color based on health
                if health_score >= 0.8:
                    color = "success"
                elif health_score >= 0.6:
                    color = "warning"
                else:
                    color = "danger"
                
                # Status icon
                if status == 'running':
                    icon = "fas fa-check-circle"
                elif status == 'starting':
                    icon = "fas fa-spinner fa-spin"
                elif status == 'stopping':
                    icon = "fas fa-stop-circle"
                else:
                    icon = "fas fa-exclamation-triangle"
                
                component_card = dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6([
                                html.I(className=icon + " me-2"),
                                component.get('component_id', 'Unknown')
                            ]),
                            html.H5(f"{health_score:.1%}", className=f"text-{color}"),
                            html.Small(status.replace('_', ' ').title())
                        ])
                    ], className="text-center")
                ], width=6, lg=4, className="mb-3")
                
                components.append(component_card)
            
            return dbc.Row(components)
        
        # Performance gauge callback
        @self.app.callback(
            Output('performance-gauge-chart', 'figure'),
            [Input('performance-data-store', 'data')]
        )
        def update_performance_gauge(performance_data):
            if not performance_data:
                return go.Figure()
            
            # Create gauge chart
            fig = go.Figure()
            
            # CPU gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data.get('cpu_usage', 0),
                domain={'x': [0, 0.5], 'y': [0.5, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ]
                }
            ))
            
            # Memory gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data.get('memory_usage', 0),
                domain={'x': [0.5, 1], 'y': [0.5, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ]
                }
            ))
            
            # Response time gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data.get('response_time', 0),
                domain={'x': [0.25, 0.75], 'y': [0, 0.5]},
                title={'text': "Response Time (ms)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgray"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "red"}
                    ]
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                font={'size': 10}
            )
            
            return fig
        
        # Alerts display callback
        @self.app.callback(
            Output('alerts-list', 'children'),
            [Input('alerts-data-store', 'data')]
        )
        def update_alerts_display(alerts_data):
            if not alerts_data:
                return html.P("No active alerts", className="text-muted")
            
            # Filter active alerts
            active_alerts = [a for a in alerts_data if a.get('status') == 'active']
            
            if not active_alerts:
                return html.P("No active alerts", className="text-success")
            
            alert_items = []
            for alert in active_alerts[:5]:  # Show top 5
                severity = alert.get('severity', 'low')
                color = {
                    'low': 'info',
                    'medium': 'warning',
                    'high': 'danger',
                    'critical': 'danger'
                }.get(severity, 'info')
                
                alert_item = dbc.Alert([
                    html.H6(alert.get('title', 'Alert')),
                    html.P(alert.get('description', ''), className="mb-0"),
                    html.Small(f"Component: {alert.get('component_id', 'Unknown')}")
                ], color=color, className="mb-2")
                
                alert_items.append(alert_item)
            
            return alert_items
        
        # System control callbacks
        @self.app.callback(
            [Output('confirmation-modal', 'is_open'),
             Output('confirmation-message', 'children'),
             Output('confirm-action-btn', 'color')],
            [Input('main-power-switch', 'value'),
             Input('emergency-stop-btn', 'n_clicks'),
             Input('cancel-action-btn', 'n_clicks')],
            [State('confirmation-modal', 'is_open'),
             State('system-data-store', 'data')]
        )
        def handle_system_control(switch_value, emergency_clicks, cancel_clicks, is_open, system_data):
            ctx = callback_context
            
            if not ctx.triggered:
                return False, "", "primary"
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'cancel-action-btn':
                return False, "", "primary"
            
            if trigger_id == 'main-power-switch':
                current_status = system_data.get('status', 'OFF') if system_data else 'OFF'
                if switch_value and current_status == 'OFF':
                    return True, "Are you sure you want to turn the system ON?", "success"
                elif not switch_value and current_status == 'ON':
                    return True, "Are you sure you want to turn the system OFF?", "warning"
            
            elif trigger_id == 'emergency-stop-btn':
                return True, "EMERGENCY STOP: This will immediately stop all system operations. Are you sure?", "danger"
            
            return False, "", "primary"
        
        # Confirm action callback
        @self.app.callback(
            [Output('success-toast', 'is_open'),
             Output('success-toast', 'children'),
             Output('alert-toast', 'is_open'),
             Output('alert-toast', 'children'),
             Output('confirmation-modal', 'is_open', allow_duplicate=True)],
            [Input('confirm-action-btn', 'n_clicks')],
            [State('confirmation-message', 'children'),
             State('action-reason', 'value'),
             State('force-action-check', 'value'),
             State('system-data-store', 'data'),
             State('main-power-switch', 'value')],
            prevent_initial_call=True
        )
        def confirm_action(n_clicks, message, reason, force_check, system_data, switch_value):
            if not n_clicks:
                return False, "", False, "", False
            
            try:
                # Determine action based on confirmation message
                if "turn the system ON" in message:
                    success = self._turn_system_on(reason, "force" in force_check)
                elif "turn the system OFF" in message:
                    success = self._turn_system_off(reason, "force" in force_check)
                elif "EMERGENCY STOP" in message:
                    success = self._emergency_stop(reason)
                else:
                    return False, "", True, "Unknown action", False
                
                if success:
                    return True, "Action completed successfully", False, "", False
                else:
                    return False, "", True, "Action failed", False
                    
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                return False, "", True, f"Error: {str(e)}", False
    
    def _setup_auth_routes(self):
        """Setup authentication routes"""
        @self.server.route('/api/auth/login', methods=['POST'])
        def api_login():
            data = flask_request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple authentication
            if username == "admin" and password == "admin123":
                session['authenticated'] = True
                session['username'] = username
                session['login_time'] = datetime.now()
                return {"success": True, "message": "Login successful"}
            else:
                return {"success": False, "message": "Invalid credentials"}, 401
        
        @self.server.route('/api/auth/logout', methods=['POST'])
        def api_logout():
            session.clear()
            return {"success": True, "message": "Logout successful"}
    
    def _fetch_system_status(self) -> Dict[str, Any]:
        """Fetch system status from API"""
        try:
            response = requests.get(
                f"{self.config.api_base_url}/api/system/status",
                headers={"Authorization": "Bearer admin-token"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching system status: {e}")
            return {}
    
    def _fetch_component_health(self) -> List[Dict[str, Any]]:
        """Fetch component health from API"""
        try:
            response = requests.get(
                f"{self.config.status_api_url}/api/status/components",
                headers={"Authorization": "Bearer admin-token"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching component health: {e}")
            return []
    
    def _fetch_alerts(self) -> List[Dict[str, Any]]:
        """Fetch alerts from API"""
        try:
            response = requests.get(
                f"{self.config.status_api_url}/api/status/alerts",
                headers={"Authorization": "Bearer admin-token"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return []
    
    def _fetch_performance_metrics(self) -> Dict[str, Any]:
        """Fetch performance metrics from API"""
        try:
            response = requests.get(
                f"{self.config.status_api_url}/api/status/performance",
                headers={"Authorization": "Bearer admin-token"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching performance metrics: {e}")
            return {}
    
    def _turn_system_on(self, reason: str, force: bool = False) -> bool:
        """Turn system on via API"""
        try:
            response = requests.post(
                f"{self.config.api_base_url}/api/system/on",
                json={
                    "action": "on",
                    "reason": reason or "Dashboard control",
                    "force": force
                },
                headers={"Authorization": "Bearer admin-token"},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except Exception as e:
            logger.error(f"Error turning system on: {e}")
            return False
    
    def _turn_system_off(self, reason: str, force: bool = False) -> bool:
        """Turn system off via API"""
        try:
            response = requests.post(
                f"{self.config.api_base_url}/api/system/off",
                json={
                    "action": "off",
                    "reason": reason or "Dashboard control",
                    "force": force
                },
                headers={"Authorization": "Bearer admin-token"},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except Exception as e:
            logger.error(f"Error turning system off: {e}")
            return False
    
    def _emergency_stop(self, reason: str) -> bool:
        """Emergency stop via API"""
        try:
            response = requests.post(
                f"{self.config.api_base_url}/api/system/emergency",
                json={
                    "action": "emergency",
                    "reason": reason or "Emergency stop from dashboard"
                },
                headers={"Authorization": "Bearer admin-token"},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except Exception as e:
            logger.error(f"Error emergency stop: {e}")
            return False
    
    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in human readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting System Control Dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

# Main execution
if __name__ == "__main__":
    config = DashboardConfig()
    dashboard = SystemControlDashboard(config)
    dashboard.run(debug=config.debug)