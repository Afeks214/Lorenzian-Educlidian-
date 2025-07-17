"""
Performance Dashboard - Agent 3

This module provides a comprehensive web-based dashboard for visualizing
performance trends, regression alerts, and system health metrics.
"""

import logging


import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from .performance_regression_system import performance_detector
from .performance_alerting_system import alerting_system
from .ci_performance_gates import ci_gates

logger = structlog.get_logger()

class PerformanceDashboard:
    """
    Web-based performance dashboard with real-time updates
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Performance Dashboard", version="1.0.0")
        self.websocket_connections = []
        
        # Setup routes
        self._setup_routes()
        
        logger.info("PerformanceDashboard initialized", 
                   host=host, port=port)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/performance/summary")
        async def get_performance_summary():
            """Get performance summary statistics"""
            return self._get_performance_summary()
        
        @self.app.get("/api/performance/trends/{test_name}")
        async def get_performance_trends(test_name: str, hours: int = 24):
            """Get performance trends for a specific test"""
            return self._get_performance_trends(test_name, hours)
        
        @self.app.get("/api/performance/alerts")
        async def get_performance_alerts():
            """Get active performance alerts"""
            return self._get_performance_alerts()
        
        @self.app.get("/api/performance/gates")
        async def get_performance_gates():
            """Get performance gate status"""
            return self._get_performance_gates()
        
        @self.app.get("/api/performance/builds")
        async def get_build_performance(limit: int = 20):
            """Get recent build performance data"""
            return self._get_build_performance(limit)
        
        @self.app.get("/api/performance/chart/{test_name}")
        async def get_performance_chart(test_name: str, hours: int = 24):
            """Get performance chart data"""
            return self._get_performance_chart(test_name, hours)
        
        @self.app.get("/api/performance/regression_analysis/{test_name}")
        async def get_regression_analysis(test_name: str):
            """Get regression analysis for a test"""
            return self._get_regression_analysis(test_name)
        
        @self.app.post("/api/performance/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge a performance alert"""
            alerting_system.acknowledge_alert(alert_id, "dashboard_user")
            return {"status": "acknowledged"}
        
        @self.app.post("/api/performance/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve a performance alert"""
            alerting_system.resolve_alert(alert_id, "dashboard_user")
            return {"status": "resolved"}
        
        @self.app.websocket("/ws/performance")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                    update_data = {
                        "type": "performance_update",
                        "data": self._get_performance_summary(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send_json(update_data)
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML content"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 10px; }
                .metric-label { color: #666; }
                .chart-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .alerts-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .alert { padding: 15px; margin: 10px 0; border-radius: 5px; }
                .alert-critical { background: #ffebee; border-left: 4px solid #f44336; }
                .alert-high { background: #fff3e0; border-left: 4px solid #ff9800; }
                .alert-medium { background: #fff8e1; border-left: 4px solid #ffc107; }
                .alert-low { background: #f3e5f5; border-left: 4px solid #9c27b0; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
                .status-ok { background: #4caf50; }
                .status-warning { background: #ff9800; }
                .status-error { background: #f44336; }
                .btn { padding: 8px 16px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; }
                .btn-primary { background: #2196f3; color: white; }
                .btn-danger { background: #f44336; color: white; }
                .btn-success { background: #4caf50; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Performance Dashboard</h1>
                    <p>Real-time performance monitoring and regression detection</p>
                    <div id="connection-status">
                        <span class="status-indicator status-ok"></span>
                        Connected
                    </div>
                </div>
                
                <div class="metrics-grid" id="metrics-grid">
                    <!-- Metrics will be populated by JavaScript -->
                </div>
                
                <div class="chart-container">
                    <h2>Performance Trends</h2>
                    <div id="trends-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Test Performance Overview</h2>
                    <div id="overview-chart"></div>
                </div>
                
                <div class="alerts-container">
                    <h2>Performance Alerts</h2>
                    <div id="alerts-container">
                        <!-- Alerts will be populated by JavaScript -->
                    </div>
                </div>
            </div>
            
            <script>
                // WebSocket connection
                const ws = new WebSocket('ws://localhost:8080/ws/performance');
                
                ws.onopen = function() {
                    console.log('Connected to performance dashboard');
                    updateConnectionStatus(true);
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'performance_update') {
                        updateDashboard(data.data);
                    }
                };
                
                ws.onclose = function() {
                    console.log('Disconnected from performance dashboard');
                    updateConnectionStatus(false);
                };
                
                function updateConnectionStatus(connected) {
                    const statusElement = document.getElementById('connection-status');
                    if (connected) {
                        statusElement.innerHTML = '<span class="status-indicator status-ok"></span>Connected';
                    } else {
                        statusElement.innerHTML = '<span class="status-indicator status-error"></span>Disconnected';
                    }
                }
                
                function updateDashboard(data) {
                    updateMetrics(data.summary);
                    updateCharts();
                    updateAlerts();
                }
                
                function updateMetrics(summary) {
                    const metricsGrid = document.getElementById('metrics-grid');
                    metricsGrid.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value">${summary.total_tests || 0}</div>
                            <div class="metric-label">Total Tests</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${summary.active_alerts || 0}</div>
                            <div class="metric-label">Active Alerts</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${summary.tests_passed || 0}</div>
                            <div class="metric-label">Tests Passed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${summary.avg_performance || 0}ms</div>
                            <div class="metric-label">Avg Performance</div>
                        </div>
                    `;
                }
                
                function updateCharts() {
                    // Update performance trends chart
                    fetch('/api/performance/chart/test_var_calculation')
                        .then(response => response.json())
                        .then(data => {
                            Plotly.newPlot('trends-chart', data.data, data.layout);
                        });
                    
                    // Update overview chart
                    fetch('/api/performance/summary')
                        .then(response => response.json())
                        .then(data => {
                            const overviewData = [{
                                x: data.test_performance.map(t => t.test_name),
                                y: data.test_performance.map(t => t.avg_time),
                                type: 'bar',
                                name: 'Average Time (ms)'
                            }];
                            
                            const overviewLayout = {
                                title: 'Test Performance Overview',
                                xaxis: { title: 'Test Name' },
                                yaxis: { title: 'Average Time (ms)' }
                            };
                            
                            Plotly.newPlot('overview-chart', overviewData, overviewLayout);
                        });
                }
                
                function updateAlerts() {
                    fetch('/api/performance/alerts')
                        .then(response => response.json())
                        .then(data => {
                            const alertsContainer = document.getElementById('alerts-container');
                            if (data.alerts.length === 0) {
                                alertsContainer.innerHTML = '<p>No active alerts</p>';
                                return;
                            }
                            
                            alertsContainer.innerHTML = data.alerts.map(alert => `
                                <div class="alert alert-${alert.severity.toLowerCase()}">
                                    <strong>${alert.test_name}</strong> - ${alert.severity}
                                    <p>${alert.message}</p>
                                    <button class="btn btn-primary" onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>
                                    <button class="btn btn-success" onclick="resolveAlert('${alert.id}')">Resolve</button>
                                </div>
                            `).join('');
                        });
                }
                
                function acknowledgeAlert(alertId) {
                    fetch(`/api/performance/alerts/${alertId}/acknowledge`, {method: 'POST'})
                        .then(() => updateAlerts());
                }
                
                function resolveAlert(alertId) {
                    fetch(`/api/performance/alerts/${alertId}/resolve`, {method: 'POST'})
                        .then(() => updateAlerts());
                }
                
                // Initial load
                updateCharts();
                updateAlerts();
                
                // Auto-refresh every 30 seconds
                setInterval(() => {
                    updateCharts();
                    updateAlerts();
                }, 30000);
            </script>
        </body>
        </html>
        """
    
    def _get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        try:
            # Get performance report
            perf_report = performance_detector.get_performance_report(24)
            
            # Get alerting report
            alert_report = alerting_system.get_alerting_report()
            
            # Get recent build performance
            build_report = self._get_build_performance_summary()
            
            return {
                "summary": {
                    "total_tests": len(perf_report.get('performance_summary', [])),
                    "active_alerts": alert_report.get('active_alerts', 0),
                    "tests_passed": sum(1 for test in perf_report.get('performance_summary', []) 
                                      if test.get('avg_time', 0) < 100),  # Assuming 100ms threshold
                    "avg_performance": sum(test.get('avg_time', 0) for test in perf_report.get('performance_summary', [])) / 
                                     max(len(perf_report.get('performance_summary', [])), 1)
                },
                "performance_report": perf_report,
                "alert_report": alert_report,
                "build_report": build_report,
                "test_performance": perf_report.get('performance_summary', [])
            }
        except Exception as e:
            logger.error("Error getting performance summary", error=str(e))
            return {"error": str(e)}
    
    def _get_performance_trends(self, test_name: str, hours: int) -> Dict:
        """Get performance trends for a specific test"""
        try:
            trends = performance_detector.get_performance_trends(test_name, hours)
            return trends
        except Exception as e:
            logger.error("Error getting performance trends", error=str(e))
            return {"error": str(e)}
    
    def _get_performance_alerts(self) -> Dict:
        """Get active performance alerts"""
        try:
            active_alerts = alerting_system.get_active_alerts()
            return {
                "alerts": [
                    {
                        "id": alert.id,
                        "test_name": alert.test_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged
                    }
                    for alert in active_alerts
                ]
            }
        except Exception as e:
            logger.error("Error getting performance alerts", error=str(e))
            return {"error": str(e)}
    
    def _get_performance_gates(self) -> Dict:
        """Get performance gate status"""
        try:
            # This would need to be implemented in ci_gates
            return {
                "gates": [
                    {
                        "name": gate.name,
                        "test_pattern": gate.test_pattern,
                        "max_time_ms": gate.max_time_ms,
                        "enabled": gate.enabled
                    }
                    for gate in ci_gates.gates.values()
                ]
            }
        except Exception as e:
            logger.error("Error getting performance gates", error=str(e))
            return {"error": str(e)}
    
    def _get_build_performance(self, limit: int) -> Dict:
        """Get recent build performance data"""
        try:
            # This would query the CI performance database
            conn = sqlite3.connect(ci_gates.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT build_id, commit_hash, branch, timestamp, environment
                FROM build_performance 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            builds = cursor.fetchall()
            conn.close()
            
            return {
                "builds": [
                    {
                        "build_id": build[0],
                        "commit_hash": build[1],
                        "branch": build[2],
                        "timestamp": build[3],
                        "environment": build[4]
                    }
                    for build in builds
                ]
            }
        except Exception as e:
            logger.error("Error getting build performance", error=str(e))
            return {"error": str(e)}
    
    def _get_build_performance_summary(self) -> Dict:
        """Get build performance summary"""
        try:
            conn = sqlite3.connect(ci_gates.db_path)
            cursor = conn.cursor()
            
            # Get recent builds count
            cursor.execute("""
                SELECT COUNT(DISTINCT build_id) 
                FROM build_performance 
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            
            recent_builds = cursor.fetchone()[0]
            
            # Get successful builds (assuming no gate failures)
            cursor.execute("""
                SELECT COUNT(DISTINCT build_id) 
                FROM build_performance b
                WHERE timestamp >= datetime('now', '-24 hours')
                AND NOT EXISTS (
                    SELECT 1 FROM gate_results g 
                    WHERE g.build_id = b.build_id AND g.passed = 0
                )
            """)
            
            successful_builds = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "recent_builds": recent_builds,
                "successful_builds": successful_builds,
                "success_rate": (successful_builds / max(recent_builds, 1)) * 100
            }
        except Exception as e:
            logger.error("Error getting build performance summary", error=str(e))
            return {"recent_builds": 0, "successful_builds": 0, "success_rate": 0}
    
    def _get_performance_chart(self, test_name: str, hours: int) -> Dict:
        """Get performance chart data"""
        try:
            conn = sqlite3.connect(performance_detector.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT timestamp, mean_time FROM performance_history 
                WHERE test_name = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (test_name, cutoff_time.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {"error": "No data found"}
            
            # Create Plotly chart
            timestamps = [datetime.fromisoformat(r[0]) for r in results]
            values = [r[1] * 1000 for r in results]  # Convert to ms
            
            trace = go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=test_name,
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            )
            
            layout = go.Layout(
                title=f'Performance Trend: {test_name}',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Response Time (ms)'),
                hovermode='closest'
            )
            
            fig = go.Figure(data=[trace], layout=layout)
            
            return {
                "data": fig.data,
                "layout": fig.layout
            }
        except Exception as e:
            logger.error("Error getting performance chart", error=str(e))
            return {"error": str(e)}
    
    def _get_regression_analysis(self, test_name: str) -> Dict:
        """Get regression analysis for a test"""
        try:
            # Get prediction data
            prediction = performance_detector.predict_performance_trend(test_name, 7)
            
            # Get recent performance data
            conn = sqlite3.connect(performance_detector.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, mean_time FROM performance_history 
                WHERE test_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (test_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {"error": "No data found"}
            
            # Calculate regression statistics
            values = [r[1] for r in results]
            recent_avg = sum(values[:10]) / 10 if len(values) >= 10 else sum(values) / len(values)
            older_avg = sum(values[-10:]) / 10 if len(values) >= 10 else recent_avg
            
            regression_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
            
            return {
                "test_name": test_name,
                "recent_average": recent_avg,
                "historical_average": older_avg,
                "regression_percent": regression_percent,
                "trend_direction": "IMPROVING" if regression_percent < 0 else "DEGRADING",
                "prediction": prediction,
                "data_points": len(results)
            }
        except Exception as e:
            logger.error("Error getting regression analysis", error=str(e))
            return {"error": str(e)}
    
    async def broadcast_update(self, data: Dict):
        """Broadcast update to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(data)
            except (ConnectionError, OSError, TimeoutError) as e:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def run(self):
        """Run the dashboard server"""
        uvicorn.run(self.app, host=self.host, port=self.port)

# Global instance
dashboard = PerformanceDashboard()