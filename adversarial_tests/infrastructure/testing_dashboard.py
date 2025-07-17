"""
Real-time Testing Dashboard

Provides real-time test monitoring, historical analysis, and performance
metrics visualization for the adversarial testing infrastructure.
"""

import asyncio
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.event_bus import EventBus
from src.core.events import Event
from .test_orchestrator import TestOrchestrator, TestStatus, TestTask, TestSession


@dataclass
class TestMetrics:
    """Test execution metrics"""
    test_id: str
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    status: str
    timestamp: datetime
    session_id: str
    error_message: Optional[str] = None


@dataclass
class SessionMetrics:
    """Session-level metrics"""
    session_id: str
    session_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_execution_time: float
    total_execution_time: float
    success_rate: float
    start_time: datetime
    end_time: Optional[datetime] = None


class MetricsDatabase:
    """SQLite database for storing test metrics"""
    
    def __init__(self, db_path: str = "test_metrics.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Test metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Session metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    session_name TEXT NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    average_execution_time REAL NOT NULL,
                    total_execution_time REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    active_sessions INTEGER NOT NULL,
                    running_tests INTEGER NOT NULL
                )
            """)
            
            self.connection.commit()
    
    def insert_test_metrics(self, metrics: TestMetrics):
        """Insert test metrics into database"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO test_metrics 
                (test_id, test_name, execution_time, memory_usage, cpu_usage, 
                 status, timestamp, session_id, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.test_id,
                metrics.test_name,
                metrics.execution_time,
                metrics.memory_usage,
                metrics.cpu_usage,
                metrics.status,
                metrics.timestamp.isoformat(),
                metrics.session_id,
                metrics.error_message
            ))
            self.connection.commit()
    
    def insert_session_metrics(self, metrics: SessionMetrics):
        """Insert session metrics into database"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO session_metrics 
                (session_id, session_name, total_tests, passed_tests, failed_tests,
                 average_execution_time, total_execution_time, success_rate,
                 start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.session_id,
                metrics.session_name,
                metrics.total_tests,
                metrics.passed_tests,
                metrics.failed_tests,
                metrics.average_execution_time,
                metrics.total_execution_time,
                metrics.success_rate,
                metrics.start_time.isoformat(),
                metrics.end_time.isoformat() if metrics.end_time else None
            ))
            self.connection.commit()
    
    def get_test_metrics(self, session_id: Optional[str] = None, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[TestMetrics]:
        """Get test metrics with optional filtering"""
        with self.lock:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM test_metrics WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                TestMetrics(
                    test_id=row[1],
                    test_name=row[2],
                    execution_time=row[3],
                    memory_usage=row[4],
                    cpu_usage=row[5],
                    status=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    session_id=row[8],
                    error_message=row[9]
                )
                for row in rows
            ]
    
    def get_session_metrics(self, limit: int = 100) -> List[SessionMetrics]:
        """Get session metrics"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM session_metrics 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            return [
                SessionMetrics(
                    session_id=row[1],
                    session_name=row[2],
                    total_tests=row[3],
                    passed_tests=row[4],
                    failed_tests=row[5],
                    average_execution_time=row[6],
                    total_execution_time=row[7],
                    success_rate=row[8],
                    start_time=datetime.fromisoformat(row[9]),
                    end_time=datetime.fromisoformat(row[10]) if row[10] else None
                )
                for row in rows
            ]


class TestingDashboard:
    """
    Real-time testing dashboard with monitoring, analytics, and visualization
    """
    
    def __init__(self, orchestrator: TestOrchestrator, port: int = 5000):
        self.orchestrator = orchestrator
        self.port = port
        self.db = MetricsDatabase()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'test-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Real-time data stores
        self.real_time_metrics = deque(maxlen=1000)
        self.active_sessions = {}
        self.system_metrics_history = deque(maxlen=100)
        
        # Event listeners
        self.orchestrator.event_bus.subscribe("session_created", self._on_session_created)
        self.orchestrator.event_bus.subscribe("task_started", self._on_task_started)
        self.orchestrator.event_bus.subscribe("task_completed", self._on_task_completed)
        self.orchestrator.event_bus.subscribe("session_completed", self._on_session_completed)
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.monitoring_task = None
        self.logger = logging.getLogger(__name__)
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/sessions')
        def get_sessions():
            """Get all sessions"""
            sessions = self.db.get_session_metrics()
            return jsonify([asdict(session) for session in sessions])
        
        @self.app.route('/api/session/<session_id>')
        def get_session(session_id):
            """Get specific session details"""
            try:
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                status = loop.run_until_complete(self.orchestrator.get_session_status(session_id))
                loop.close()
                return jsonify(status)
            except Exception as e:
                return jsonify({"error": str(e)}), 404
        
        @self.app.route('/api/metrics/test')
        def get_test_metrics():
            """Get test metrics with filtering"""
            session_id = request.args.get('session_id')
            start_time = request.args.get('start_time')
            end_time = request.args.get('end_time')
            
            start_dt = datetime.fromisoformat(start_time) if start_time else None
            end_dt = datetime.fromisoformat(end_time) if end_time else None
            
            metrics = self.db.get_test_metrics(session_id, start_dt, end_dt)
            return jsonify([asdict(metric) for metric in metrics])
        
        @self.app.route('/api/metrics/system')
        def get_system_metrics():
            """Get current system metrics"""
            metrics = asyncio.run(self.orchestrator.get_system_metrics())
            return jsonify(metrics)
        
        @self.app.route('/api/analytics/performance')
        def get_performance_analytics():
            """Get performance analytics"""
            analytics = self._generate_performance_analytics()
            return jsonify(analytics)
        
        @self.app.route('/api/analytics/trends')
        def get_trend_analytics():
            """Get trend analytics"""
            trends = self._generate_trend_analytics()
            return jsonify(trends)
        
        @self.app.route('/api/charts/execution_time')
        def get_execution_time_chart():
            """Get execution time chart data"""
            chart_data = self._generate_execution_time_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/success_rate')
        def get_success_rate_chart():
            """Get success rate chart data"""
            chart_data = self._generate_success_rate_chart()
            return jsonify(chart_data)
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Client connected to dashboard")
            emit('connected', {'message': 'Connected to test dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_live_data')
        def handle_live_data_request():
            """Handle live data request"""
            self._send_live_data()
    
    async def _on_session_created(self, event: Event):
        """Handle session created event"""
        session_data = event.data
        self.active_sessions[session_data['session_id']] = {
            'name': session_data['name'],
            'start_time': session_data['timestamp'],
            'tests': {},
            'metrics': {
                'total_tests': 0,
                'completed_tests': 0,
                'failed_tests': 0,
                'running_tests': 0
            }
        }
        
        # Emit to connected clients
        self.socketio.emit('session_created', session_data)
    
    async def _on_task_started(self, event: Event):
        """Handle task started event"""
        task_data = event.data
        session_id = task_data['session_id']
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['tests'][task_data['test_id']] = {
                'name': task_data['test_name'],
                'start_time': task_data['timestamp'],
                'status': 'running'
            }
            self.active_sessions[session_id]['metrics']['running_tests'] += 1
        
        # Emit to connected clients
        self.socketio.emit('task_started', task_data)
    
    async def _on_task_completed(self, event: Event):
        """Handle task completed event"""
        task_data = event.data
        session_id = task_data['session_id']
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Update test info
            if task_data['test_id'] in session['tests']:
                session['tests'][task_data['test_id']]['status'] = task_data['status']
                session['tests'][task_data['test_id']]['end_time'] = task_data['timestamp']
                session['tests'][task_data['test_id']]['execution_time'] = task_data['execution_time']
            
            # Update metrics
            session['metrics']['completed_tests'] += 1
            session['metrics']['running_tests'] -= 1
            
            if task_data['status'] == 'failed':
                session['metrics']['failed_tests'] += 1
        
        # Create and store test metrics
        test_metrics = TestMetrics(
            test_id=task_data['test_id'],
            test_name=task_data.get('test_name', 'Unknown'),
            execution_time=task_data['execution_time'],
            memory_usage=task_data.get('memory_usage', 0.0),
            cpu_usage=task_data.get('cpu_usage', 0.0),
            status=task_data['status'],
            timestamp=datetime.fromisoformat(task_data['timestamp']),
            session_id=session_id,
            error_message=task_data.get('error')
        )
        
        self.db.insert_test_metrics(test_metrics)
        self.real_time_metrics.append(test_metrics)
        
        # Emit to connected clients
        self.socketio.emit('task_completed', task_data)
        self._send_live_data()
    
    async def _on_session_completed(self, event: Event):
        """Handle session completed event"""
        session_data = event.data
        session_id = session_data['session_id']
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            metrics_data = session_data['metrics']
            
            # Create and store session metrics
            session_metrics = SessionMetrics(
                session_id=session_id,
                session_name=session['name'],
                total_tests=metrics_data['total_tests'],
                passed_tests=metrics_data['passed_tests'],
                failed_tests=metrics_data['failed_tests'],
                average_execution_time=metrics_data['average_execution_time'],
                total_execution_time=metrics_data['total_execution_time'],
                success_rate=metrics_data['success_rate'],
                start_time=datetime.fromisoformat(session['start_time']),
                end_time=datetime.fromisoformat(session_data['timestamp'])
            )
            
            self.db.insert_session_metrics(session_metrics)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
        
        # Emit to connected clients
        self.socketio.emit('session_completed', session_data)
        self._send_live_data()
    
    def _send_live_data(self):
        """Send live data to connected clients"""
        live_data = {
            'active_sessions': len(self.active_sessions),
            'recent_metrics': [asdict(m) for m in list(self.real_time_metrics)[-10:]],
            'system_metrics': self.system_metrics_history[-1] if self.system_metrics_history else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.socketio.emit('live_data', live_data)
    
    def _generate_performance_analytics(self) -> Dict:
        """Generate performance analytics"""
        # Get recent test metrics
        recent_metrics = self.db.get_test_metrics(
            start_time=datetime.now() - timedelta(hours=24)
        )
        
        if not recent_metrics:
            return {"error": "No data available"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in recent_metrics]
        success_rate = len([m for m in recent_metrics if m.status == 'completed']) / len(recent_metrics) * 100
        
        analytics = {
            'total_tests': len(recent_metrics),
            'average_execution_time': np.mean(execution_times),
            'median_execution_time': np.median(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'success_rate': success_rate,
            'failure_rate': 100 - success_rate,
            'total_execution_time': sum(execution_times)
        }
        
        return analytics
    
    def _generate_trend_analytics(self) -> Dict:
        """Generate trend analytics"""
        # Get metrics for last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        metrics = self.db.get_test_metrics(start_time=start_time, end_time=end_time)
        
        if not metrics:
            return {"error": "No data available"}
        
        # Group by day
        daily_metrics = defaultdict(list)
        for metric in metrics:
            day = metric.timestamp.date()
            daily_metrics[day].append(metric)
        
        # Calculate daily trends
        trends = []
        for day, day_metrics in sorted(daily_metrics.items()):
            success_rate = len([m for m in day_metrics if m.status == 'completed']) / len(day_metrics) * 100
            avg_execution_time = np.mean([m.execution_time for m in day_metrics])
            
            trends.append({
                'date': day.isoformat(),
                'total_tests': len(day_metrics),
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time
            })
        
        return {'trends': trends}
    
    def _generate_execution_time_chart(self) -> Dict:
        """Generate execution time chart data"""
        metrics = self.db.get_test_metrics(
            start_time=datetime.now() - timedelta(hours=24)
        )
        
        if not metrics:
            return {"error": "No data available"}
        
        # Create histogram data
        execution_times = [m.execution_time for m in metrics]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=execution_times,
            nbinsx=20,
            name='Execution Time Distribution'
        ))
        
        fig.update_layout(
            title='Test Execution Time Distribution',
            xaxis_title='Execution Time (seconds)',
            yaxis_title='Count'
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    def _generate_success_rate_chart(self) -> Dict:
        """Generate success rate chart data"""
        sessions = self.db.get_session_metrics(limit=20)
        
        if not sessions:
            return {"error": "No data available"}
        
        # Extract data for chart
        session_names = [s.session_name for s in sessions]
        success_rates = [s.success_rate for s in sessions]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=session_names,
            y=success_rates,
            name='Success Rate'
        ))
        
        fig.update_layout(
            title='Session Success Rates',
            xaxis_title='Session',
            yaxis_title='Success Rate (%)'
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    async def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitor_system())
    
    async def _monitor_system(self):
        """Monitor system metrics"""
        while True:
            try:
                # Get system metrics
                metrics = await self.orchestrator.get_system_metrics()
                
                # Store in history
                self.system_metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                
                # Send to clients
                self.socketio.emit('system_metrics', metrics)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring system: {e}")
                await asyncio.sleep(10)
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.logger.info(f"Starting testing dashboard on port {self.port}")
        
        # Start monitoring in background
        asyncio.create_task(self.start_monitoring())
        
        # Run Flask app
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)
    
    def generate_report(self, session_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict:
        """Generate comprehensive test report"""
        # Get test metrics
        test_metrics = self.db.get_test_metrics(session_id, start_time, end_time)
        
        if not test_metrics:
            return {"error": "No data available for report"}
        
        # Calculate statistics
        total_tests = len(test_metrics)
        passed_tests = len([m for m in test_metrics if m.status == 'completed'])
        failed_tests = len([m for m in test_metrics if m.status == 'failed'])
        
        execution_times = [m.execution_time for m in test_metrics]
        
        # Generate report
        report = {
            'report_generated': datetime.now().isoformat(),
            'filter_criteria': {
                'session_id': session_id,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None
            },
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_execution_time': sum(execution_times),
                'average_execution_time': np.mean(execution_times),
                'median_execution_time': np.median(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times)
            },
            'test_details': [asdict(m) for m in test_metrics],
            'failure_analysis': {
                'failed_tests': [asdict(m) for m in test_metrics if m.status == 'failed'],
                'common_errors': self._analyze_common_errors(test_metrics)
            }
        }
        
        return report
    
    def _analyze_common_errors(self, metrics: List[TestMetrics]) -> Dict:
        """Analyze common error patterns"""
        failed_metrics = [m for m in metrics if m.status == 'failed' and m.error_message]
        
        if not failed_metrics:
            return {"error": "No failed tests with error messages"}
        
        # Count error types
        error_counts = defaultdict(int)
        for metric in failed_metrics:
            error_type = metric.error_message.split(':')[0] if ':' in metric.error_message else metric.error_message
            error_counts[error_type] += 1
        
        # Sort by frequency
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_failed_tests': len(failed_metrics),
            'unique_error_types': len(error_counts),
            'most_common_errors': common_errors[:10]
        }


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Adversarial Testing Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metric-card { 
            background: #f5f5f5; 
            padding: 20px; 
            margin: 10px; 
            border-radius: 8px;
            display: inline-block;
            min-width: 200px;
        }
        .chart-container { margin: 20px 0; }
        .live-indicator { 
            background: #4CAF50; 
            color: white; 
            padding: 5px 10px; 
            border-radius: 4px; 
            font-size: 12px;
        }
        .failed { background: #f44336; color: white; }
        .completed { background: #4CAF50; color: white; }
        .running { background: #ff9800; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Adversarial Testing Dashboard <span class="live-indicator" id="status">LIVE</span></h1>
        
        <div id="metrics">
            <div class="metric-card">
                <h3>Active Sessions</h3>
                <div id="active-sessions">0</div>
            </div>
            <div class="metric-card">
                <h3>Recent Tests</h3>
                <div id="recent-tests">0</div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div id="success-rate">0%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="execution-time-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="success-rate-chart"></div>
        </div>
        
        <div id="live-tests">
            <h3>Live Test Activity</h3>
            <div id="test-activity"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to dashboard');
            socket.emit('request_live_data');
        });
        
        socket.on('live_data', function(data) {
            updateMetrics(data);
        });
        
        socket.on('task_started', function(data) {
            addTestActivity(data, 'started');
        });
        
        socket.on('task_completed', function(data) {
            addTestActivity(data, 'completed');
        });
        
        function updateMetrics(data) {
            document.getElementById('active-sessions').textContent = data.active_sessions;
            document.getElementById('recent-tests').textContent = data.recent_metrics.length;
            
            if (data.recent_metrics.length > 0) {
                const completed = data.recent_metrics.filter(m => m.status === 'completed').length;
                const successRate = (completed / data.recent_metrics.length * 100).toFixed(1);
                document.getElementById('success-rate').textContent = successRate + '%';
            }
        }
        
        function addTestActivity(data, action) {
            const activity = document.getElementById('test-activity');
            const div = document.createElement('div');
            div.className = `activity-item ${data.status || action}`;
            div.innerHTML = `
                <span class="time">${new Date().toLocaleTimeString()}</span>
                <span class="test-name">${data.test_name}</span>
                <span class="status ${data.status || action}">${data.status || action}</span>
            `;
            activity.insertBefore(div, activity.firstChild);
            
            // Keep only last 10 activities
            while (activity.children.length > 10) {
                activity.removeChild(activity.lastChild);
            }
        }
        
        // Load charts
        fetch('/api/charts/execution_time')
            .then(response => response.json())
            .then(data => {
                if (!data.error) {
                    Plotly.newPlot('execution-time-chart', data.data, data.layout);
                }
            });
        
        fetch('/api/charts/success_rate')
            .then(response => response.json())
            .then(data => {
                if (!data.error) {
                    Plotly.newPlot('success-rate-chart', data.data, data.layout);
                }
            });
    </script>
</body>
</html>
"""

# Save HTML template
def create_dashboard_template():
    """Create the dashboard HTML template"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
        f.write(DASHBOARD_HTML)


# Demo function
async def demo_dashboard():
    """Demonstration of the testing dashboard"""
    from .test_orchestrator import TestOrchestrator, TestTask, TestPriority, example_test_function
    
    # Create orchestrator
    orchestrator = TestOrchestrator(max_parallel_tests=3)
    
    # Create dashboard
    dashboard = TestingDashboard(orchestrator, port=5001)
    
    # Create dashboard template
    create_dashboard_template()
    
    # Start dashboard in background
    dashboard_task = asyncio.create_task(dashboard.start_monitoring())
    
    # Create and run test session
    session_id = await orchestrator.create_session("Dashboard Demo Session")
    
    # Add test tasks
    tasks = [
        TestTask(
            test_id=f"test_{i}",
            test_name=f"Test {i}",
            test_function=example_test_function,
            args=(i * 0.5, i % 3 == 0),  # Some tests will fail
            priority=TestPriority.HIGH if i < 3 else TestPriority.MEDIUM
        )
        for i in range(10)
    ]
    
    for task in tasks:
        await orchestrator.add_test_task(session_id, task)
    
    # Execute session
    results = await orchestrator.execute_session(session_id)
    
    # Generate report
    report = dashboard.generate_report(session_id)
    
    print("\n=== TESTING DASHBOARD DEMO ===")
    print(f"Session Results: {len(results['results'])} tests executed")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average Execution Time: {report['summary']['average_execution_time']:.2f}s")
    print(f"Dashboard running on http://localhost:5001")
    
    # Keep dashboard running
    try:
        await dashboard_task
    except KeyboardInterrupt:
        print("\nDashboard stopped")


if __name__ == "__main__":
    create_dashboard_template()
    asyncio.run(demo_dashboard())