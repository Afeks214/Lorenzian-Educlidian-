"""
Real-time RTO Dashboard for monitoring Database and Trading Engine recovery objectives.

This module provides a comprehensive dashboard for RTO monitoring with:
- Real-time RTO metrics visualization
- Historical trend analysis
- Breach alerting and notifications
- Performance analytics
- Interactive controls for testing
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import sqlite3
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.monitoring.rto_monitor import RTOMonitoringSystem, RTOStatus, RTOMetric
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class RTODashboard:
    """Real-time RTO monitoring dashboard."""
    
    def __init__(self, rto_monitor: RTOMonitoringSystem):
        self.rto_monitor = rto_monitor
        self.app = FastAPI(title="RTO Monitoring Dashboard")
        self.active_connections: List[WebSocket] = []
        self.event_bus = rto_monitor.event_bus
        
        # Setup routes
        self._setup_routes()
        
        # Subscribe to RTO events
        self.event_bus.subscribe("rto_breach_alert", self._handle_breach_alert)
        self.event_bus.subscribe("rto_recovery_alert", self._handle_recovery_alert)
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve the main dashboard page."""
            return HTMLResponse(content=self._get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Send initial data
                await self._send_initial_data(websocket)
                
                # Keep connection alive and handle messages
                while True:
                    data = await websocket.receive_text()
                    await self._handle_websocket_message(websocket, data)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
        
        @self.app.get("/api/rto/summary")
        async def get_rto_summary(hours: int = 24):
            """Get RTO summary data."""
            return self.rto_monitor.get_rto_summary(hours)
        
        @self.app.get("/api/rto/trends/{component}")
        async def get_rto_trends(component: str, days: int = 7):
            """Get RTO trends for a component."""
            return self.rto_monitor.get_historical_trends(component, days)
        
        @self.app.get("/api/rto/metrics/{component}")
        async def get_rto_metrics(component: str, hours: int = 24):
            """Get recent RTO metrics for a component."""
            return self.rto_monitor.database.get_recent_metrics(component, hours)
        
        @self.app.post("/api/rto/test/{component}")
        async def test_rto_scenario(component: str, scenario: str = "default"):
            """Trigger RTO test scenario."""
            try:
                metric = await self.rto_monitor.simulate_failure_recovery(component, scenario)
                return {"status": "success", "metric": metric.to_dict()}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/api/rto/status")
        async def get_rto_status():
            """Get current RTO status."""
            summary = self.rto_monitor.get_rto_summary(1)  # Last hour
            
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "components": {}
            }
            
            for component, data in summary.items():
                breach_count = data.get("breach_count", 0)
                avg_rto = data.get("average_rto", 0)
                target_rto = data.get("target_rto", 0)
                
                if breach_count > 0:
                    comp_status = "breach"
                elif avg_rto > target_rto * 0.8:
                    comp_status = "warning"
                else:
                    comp_status = "healthy"
                
                status["components"][component] = {
                    "status": comp_status,
                    "current_rto": avg_rto,
                    "target_rto": target_rto,
                    "breach_count": breach_count,
                    "availability": data.get("availability_percentage", 100)
                }
                
                # Update overall status
                if comp_status == "breach":
                    status["overall_status"] = "breach"
                elif comp_status == "warning" and status["overall_status"] == "healthy":
                    status["overall_status"] = "warning"
            
            return status
    
    async def _send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data."""
        summary = self.rto_monitor.get_rto_summary(24)
        await websocket.send_text(json.dumps({
            "type": "initial_data",
            "data": summary
        }))
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "get_trends":
                component = data.get("component")
                days = data.get("days", 7)
                trends = self.rto_monitor.get_historical_trends(component, days)
                
                await websocket.send_text(json.dumps({
                    "type": "trends_data",
                    "component": component,
                    "data": trends
                }))
            
            elif msg_type == "test_scenario":
                component = data.get("component")
                scenario = data.get("scenario", "default")
                
                # Run test in background
                asyncio.create_task(self._run_test_scenario(component, scenario))
                
                await websocket.send_text(json.dumps({
                    "type": "test_started",
                    "component": component,
                    "scenario": scenario
                }))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _run_test_scenario(self, component: str, scenario: str):
        """Run RTO test scenario and broadcast results."""
        try:
            metric = await self.rto_monitor.simulate_failure_recovery(component, scenario)
            
            # Broadcast test results
            await self._broadcast_to_all({
                "type": "test_result",
                "component": component,
                "scenario": scenario,
                "metric": metric.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Error running test scenario: {e}")
            await self._broadcast_to_all({
                "type": "test_error",
                "component": component,
                "scenario": scenario,
                "error": str(e)
            })
    
    async def _handle_breach_alert(self, alert_data: Dict[str, Any]):
        """Handle RTO breach alert."""
        await self._broadcast_to_all({
            "type": "breach_alert",
            "data": alert_data
        })
    
    async def _handle_recovery_alert(self, alert_data: Dict[str, Any]):
        """Handle RTO recovery alert."""
        await self._broadcast_to_all({
            "type": "recovery_alert",
            "data": alert_data
        })
    
    async def _broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>RTO Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .status {
            padding: 0.5rem 1rem;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            text-align: center;
            margin: 0.5rem 0;
        }
        .status.healthy { background: #27ae60; }
        .status.warning { background: #f39c12; }
        .status.breach { background: #e74c3c; }
        .status.critical { background: #c0392b; }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: #ecf0f1;
            border-radius: 4px;
        }
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        .chart-container {
            height: 300px;
            margin: 1rem 0;
        }
        .controls {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .btn-primary {
            background: #3498db;
            color: white;
        }
        .btn-warning {
            background: #f39c12;
            color: white;
        }
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .alert {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            animation: slideIn 0.3s ease-out;
        }
        .alert-danger {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .alert-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .trend-chart {
            background: #ecf0f1;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            color: white;
            font-size: 0.8rem;
        }
        .connected { background: #27ae60; }
        .disconnected { background: #e74c3c; }
        .loading {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RTO Monitoring Dashboard</h1>
        <p>Real-time Recovery Time Objective monitoring for Database (&lt;30s) and Trading Engine (&lt;5s)</p>
    </div>
    
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="container">
        <div id="alerts"></div>
        
        <div class="grid">
            <div class="card">
                <h3>Database RTO Status</h3>
                <div id="databaseStatus" class="loading">Loading...</div>
                <div id="databaseMetrics"></div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="testScenario('database', 'connection_loss')">Test Connection Loss</button>
                    <button class="btn btn-warning" onclick="testScenario('database', 'disk_full')">Test Disk Full</button>
                    <button class="btn btn-danger" onclick="testScenario('database', 'primary_failure')">Test Primary Failure</button>
                </div>
            </div>
            
            <div class="card">
                <h3>Trading Engine RTO Status</h3>
                <div id="tradingStatus" class="loading">Loading...</div>
                <div id="tradingMetrics"></div>
                <div class="controls">
                    <button class="btn btn-primary" onclick="testScenario('trading_engine', 'service_crash')">Test Service Crash</button>
                    <button class="btn btn-warning" onclick="testScenario('trading_engine', 'memory_leak')">Test Memory Leak</button>
                    <button class="btn btn-danger" onclick="testScenario('trading_engine', 'config_error')">Test Config Error</button>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>RTO Trends (24 Hours)</h3>
            <div id="trendsChart" class="chart-container">
                <div class="trend-chart">
                    <p>Real-time trend chart would be displayed here using Chart.js or similar visualization library.</p>
                    <p>This would show RTO measurements over time for both components with breach indicators.</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>System Overview</h3>
            <div id="systemOverview" class="loading">Loading...</div>
        </div>
    </div>
    
    <script>
        class RTODashboard {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 5000;
                this.connect();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                this.ws.onopen = () => {
                    console.log('Connected to RTO dashboard');
                    this.updateConnectionStatus(true);
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from RTO dashboard');
                    this.updateConnectionStatus(false);
                    setTimeout(() => this.connect(), this.reconnectInterval);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connectionStatus');
                statusEl.textContent = connected ? 'Connected' : 'Disconnected';
                statusEl.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            }
            
            handleMessage(data) {
                switch(data.type) {
                    case 'initial_data':
                        this.updateDashboard(data.data);
                        break;
                    case 'breach_alert':
                        this.showAlert(data.data, 'danger');
                        break;
                    case 'recovery_alert':
                        this.showAlert(data.data, 'success');
                        break;
                    case 'test_started':
                        this.showAlert({alert_type: 'test_started', component: data.component, scenario: data.scenario}, 'info');
                        break;
                    case 'test_result':
                        this.showAlert({alert_type: 'test_completed', component: data.component, metric: data.metric}, 'success');
                        break;
                    case 'test_error':
                        this.showAlert({alert_type: 'test_error', component: data.component, error: data.error}, 'danger');
                        break;
                }
            }
            
            updateDashboard(data) {
                // Update database status
                this.updateComponentStatus('database', data.database);
                
                // Update trading engine status
                this.updateComponentStatus('trading_engine', data.trading_engine);
                
                // Update system overview
                this.updateSystemOverview(data);
            }
            
            updateComponentStatus(component, data) {
                const statusEl = document.getElementById(`${component === 'database' ? 'database' : 'trading'}Status`);
                const metricsEl = document.getElementById(`${component === 'database' ? 'database' : 'trading'}Metrics`);
                
                if (!data) return;
                
                const statusClass = data.breach_count > 0 ? 'breach' : 
                                   data.average_rto > data.target_rto * 0.8 ? 'warning' : 'healthy';
                
                statusEl.innerHTML = `
                    <div class="status ${statusClass}">
                        ${statusClass.toUpperCase()}
                    </div>
                `;
                
                metricsEl.innerHTML = `
                    <div class="metric">
                        <span>Target RTO:</span>
                        <span class="metric-value">${data.target_rto}s</span>
                    </div>
                    <div class="metric">
                        <span>Average RTO:</span>
                        <span class="metric-value">${data.average_rto.toFixed(2)}s</span>
                    </div>
                    <div class="metric">
                        <span>Breaches (24h):</span>
                        <span class="metric-value">${data.breach_count}</span>
                    </div>
                    <div class="metric">
                        <span>Availability:</span>
                        <span class="metric-value">${data.availability_percentage.toFixed(1)}%</span>
                    </div>
                `;
            }
            
            updateSystemOverview(data) {
                const overviewEl = document.getElementById('systemOverview');
                const totalMeasurements = (data.database?.total_measurements || 0) + (data.trading_engine?.total_measurements || 0);
                const totalBreaches = (data.database?.breach_count || 0) + (data.trading_engine?.breach_count || 0);
                const overallAvailability = totalMeasurements > 0 ? 
                    ((totalMeasurements - totalBreaches) / totalMeasurements * 100) : 100;
                
                overviewEl.innerHTML = `
                    <div class="metric">
                        <span>Overall Availability:</span>
                        <span class="metric-value">${overallAvailability.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>Total Measurements:</span>
                        <span class="metric-value">${totalMeasurements}</span>
                    </div>
                    <div class="metric">
                        <span>Total Breaches:</span>
                        <span class="metric-value">${totalBreaches}</span>
                    </div>
                    <div class="metric">
                        <span>Last Updated:</span>
                        <span class="metric-value">${new Date().toLocaleTimeString()}</span>
                    </div>
                `;
            }
            
            showAlert(alertData, type) {
                const alertsEl = document.getElementById('alerts');
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${type}`;
                
                let message = '';
                if (alertData.alert_type === 'rto_breach') {
                    message = `RTO BREACH: ${alertData.component} - Target: ${alertData.target_rto}s, Actual: ${alertData.actual_rto}s (${alertData.breach_percentage.toFixed(1)}% breach)`;
                } else if (alertData.alert_type === 'rto_recovery') {
                    message = `RTO RECOVERY: ${alertData.component} - Recovered in ${alertData.recovery_time}s`;
                } else if (alertData.alert_type === 'test_started') {
                    message = `Test started: ${alertData.component} - ${alertData.scenario}`;
                } else if (alertData.alert_type === 'test_completed') {
                    message = `Test completed: ${alertData.component} - RTO: ${alertData.metric.actual_seconds.toFixed(2)}s`;
                } else if (alertData.alert_type === 'test_error') {
                    message = `Test error: ${alertData.component} - ${alertData.error}`;
                }
                
                alertDiv.innerHTML = `
                    <strong>${new Date().toLocaleTimeString()}</strong> - ${message}
                    <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; font-size: 1.2em; cursor: pointer;">&times;</button>
                `;
                
                alertsEl.insertBefore(alertDiv, alertsEl.firstChild);
                
                // Auto-remove after 10 seconds
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.remove();
                    }
                }, 10000);
            }
            
            sendMessage(message) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify(message));
                }
            }
        }
        
        function testScenario(component, scenario) {
            dashboard.sendMessage({
                type: 'test_scenario',
                component: component,
                scenario: scenario
            });
        }
        
        // Initialize dashboard
        const dashboard = new RTODashboard();
        
        // Update dashboard every 30 seconds
        setInterval(() => {
            dashboard.sendMessage({type: 'refresh'});
        }, 30000);
    </script>
</body>
</html>
        """
    
    async def start(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the dashboard server."""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

async def create_rto_dashboard(rto_monitor: RTOMonitoringSystem, host: str = "0.0.0.0", port: int = 8001):
    """Create and start RTO dashboard."""
    dashboard = RTODashboard(rto_monitor)
    await dashboard.start(host, port)