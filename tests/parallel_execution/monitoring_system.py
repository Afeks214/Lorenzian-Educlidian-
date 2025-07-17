"""
Real-time Test Execution Monitoring and Worker Health Tracking System
Agent 2 Mission: Advanced Monitoring and Health Management

This module provides comprehensive real-time monitoring of test execution,
worker health tracking, automatic recovery mechanisms, and performance dashboards.
"""

import asyncio
import websockets
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import os
import signal
import subprocess
from enum import Enum
import sqlite3
from pathlib import Path
import queue
import socket
import hashlib

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    TERMINATED = "terminated"
    RECOVERING = "recovering"


@dataclass
class WorkerHealth:
    """Worker health metrics"""
    worker_id: str
    status: WorkerStatus
    process_id: int
    cpu_usage: float
    memory_usage: float
    tests_executed: int
    tests_passed: int
    tests_failed: int
    last_heartbeat: datetime
    response_time: float
    error_rate: float
    recovery_attempts: int = 0
    uptime_seconds: float = 0
    performance_score: float = 100.0
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        now = datetime.now()
        heartbeat_age = (now - self.last_heartbeat).total_seconds()
        
        return (
            self.status not in [WorkerStatus.FAILED, WorkerStatus.TERMINATED] and
            heartbeat_age < 30 and  # Heartbeat within 30 seconds
            self.error_rate < 0.1 and  # Less than 10% error rate
            self.response_time < 5.0 and  # Response time under 5 seconds
            self.performance_score > 50.0  # Performance score above 50%
        )


@dataclass
class TestExecutionEvent:
    """Test execution event"""
    event_type: str  # started, completed, failed, skipped
    test_name: str
    worker_id: str
    timestamp: datetime
    duration: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: datetime
    total_workers: int
    active_workers: int
    healthy_workers: int
    total_tests_running: int
    tests_completed: int
    tests_failed: int
    avg_test_duration: float
    system_cpu_usage: float
    system_memory_usage: float
    overall_performance_score: float


class WorkerHealthTracker:
    """Track individual worker health and performance"""
    
    def __init__(self):
        self.workers: Dict[str, WorkerHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_lock = threading.Lock()
        self.recovery_callbacks: List[Callable] = []
        
    def register_worker(self, worker_id: str, process_id: int):
        """Register a new worker"""
        with self.health_lock:
            self.workers[worker_id] = WorkerHealth(
                worker_id=worker_id,
                status=WorkerStatus.INITIALIZING,
                process_id=process_id,
                cpu_usage=0.0,
                memory_usage=0.0,
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                last_heartbeat=datetime.now(),
                response_time=0.0,
                error_rate=0.0
            )
            logger.info(f"Registered worker {worker_id} with PID {process_id}")
    
    def update_worker_health(self, worker_id: str, metrics: Dict[str, Any]):
        """Update worker health metrics"""
        with self.health_lock:
            if worker_id not in self.workers:
                logger.warning(f"Received metrics for unknown worker {worker_id}")
                return
            
            worker = self.workers[worker_id]
            
            # Update metrics
            worker.cpu_usage = metrics.get('cpu_usage', worker.cpu_usage)
            worker.memory_usage = metrics.get('memory_usage', worker.memory_usage)
            worker.tests_executed = metrics.get('tests_executed', worker.tests_executed)
            worker.tests_passed = metrics.get('tests_passed', worker.tests_passed)
            worker.tests_failed = metrics.get('tests_failed', worker.tests_failed)
            worker.response_time = metrics.get('response_time', worker.response_time)
            worker.last_heartbeat = datetime.now()
            
            # Calculate error rate
            if worker.tests_executed > 0:
                worker.error_rate = worker.tests_failed / worker.tests_executed
            
            # Update performance score
            worker.performance_score = self._calculate_performance_score(worker)
            
            # Update status based on health
            worker.status = self._determine_worker_status(worker)
            
            # Store in history
            self.health_history[worker_id].append(asdict(worker))
            
            # Check for recovery needs
            if not worker.is_healthy():
                self._trigger_recovery(worker_id)
    
    def _calculate_performance_score(self, worker: WorkerHealth) -> float:
        """Calculate worker performance score (0-100)"""
        score = 100.0
        
        # Reduce score for high error rate
        score -= worker.error_rate * 50
        
        # Reduce score for slow response time
        if worker.response_time > 1.0:
            score -= min(25, (worker.response_time - 1.0) * 10)
        
        # Reduce score for high resource usage
        if worker.cpu_usage > 80:
            score -= (worker.cpu_usage - 80) * 0.5
        
        if worker.memory_usage > 1000:  # > 1GB
            score -= min(20, (worker.memory_usage - 1000) / 100)
        
        return max(0, score)
    
    def _determine_worker_status(self, worker: WorkerHealth) -> WorkerStatus:
        """Determine worker status based on health metrics"""
        if not worker.is_healthy():
            if worker.error_rate > 0.5:
                return WorkerStatus.FAILED
            elif worker.cpu_usage > 95 or worker.memory_usage > 2000:
                return WorkerStatus.OVERLOADED
            else:
                return WorkerStatus.RUNNING
        
        return WorkerStatus.IDLE if worker.tests_executed == 0 else WorkerStatus.RUNNING
    
    def _trigger_recovery(self, worker_id: str):
        """Trigger recovery for unhealthy worker"""
        with self.health_lock:
            worker = self.workers[worker_id]
            worker.recovery_attempts += 1
            worker.status = WorkerStatus.RECOVERING
            
            # Call recovery callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback(worker_id, worker)
                except Exception as e:
                    logger.error(f"Recovery callback failed for worker {worker_id}: {e}")
    
    def get_worker_health(self, worker_id: str) -> Optional[WorkerHealth]:
        """Get current health of a worker"""
        with self.health_lock:
            return self.workers.get(worker_id)
    
    def get_all_workers_health(self) -> List[WorkerHealth]:
        """Get health of all workers"""
        with self.health_lock:
            return list(self.workers.values())
    
    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs"""
        with self.health_lock:
            return [
                worker_id for worker_id, worker in self.workers.items()
                if worker.is_healthy()
            ]
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from tracking"""
        with self.health_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker {worker_id} from tracking")
    
    def add_recovery_callback(self, callback: Callable):
        """Add recovery callback function"""
        self.recovery_callbacks.append(callback)


class TestExecutionMonitor:
    """Monitor test execution events and metrics"""
    
    def __init__(self):
        self.events: deque = deque(maxlen=10000)
        self.active_tests: Dict[str, TestExecutionEvent] = {}
        self.test_metrics: Dict[str, List[float]] = defaultdict(list)
        self.events_lock = threading.Lock()
        self.subscribers: List[Callable] = []
        
    def record_event(self, event: TestExecutionEvent):
        """Record a test execution event"""
        with self.events_lock:
            self.events.append(event)
            
            if event.event_type == "started":
                self.active_tests[event.test_name] = event
            elif event.event_type in ["completed", "failed", "skipped"]:
                if event.test_name in self.active_tests:
                    del self.active_tests[event.test_name]
                
                # Record duration metrics
                if event.duration:
                    self.test_metrics[event.test_name].append(event.duration)
                    # Keep only last 100 measurements
                    if len(self.test_metrics[event.test_name]) > 100:
                        self.test_metrics[event.test_name].pop(0)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    logger.error(f"Event subscriber failed: {e}")
    
    def get_recent_events(self, limit: int = 100) -> List[TestExecutionEvent]:
        """Get recent test execution events"""
        with self.events_lock:
            return list(self.events)[-limit:]
    
    def get_active_tests(self) -> Dict[str, TestExecutionEvent]:
        """Get currently active tests"""
        with self.events_lock:
            return self.active_tests.copy()
    
    def get_test_statistics(self, test_name: str) -> Dict[str, Any]:
        """Get statistics for a specific test"""
        with self.events_lock:
            if test_name not in self.test_metrics:
                return {"error": "No data available"}
            
            durations = self.test_metrics[test_name]
            return {
                "test_name": test_name,
                "execution_count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "recent_duration": durations[-1] if durations else 0
            }
    
    def add_subscriber(self, subscriber: Callable):
        """Add event subscriber"""
        self.subscribers.append(subscriber)


class RealTimeMonitoringSystem:
    """Comprehensive real-time monitoring system"""
    
    def __init__(self, websocket_port: int = 8765):
        self.worker_tracker = WorkerHealthTracker()
        self.execution_monitor = TestExecutionMonitor()
        self.websocket_port = websocket_port
        self.websocket_server = None
        self.monitoring_active = False
        self.connected_clients: set = set()
        self.monitoring_thread = None
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Setup recovery callbacks
        self.worker_tracker.add_recovery_callback(self._handle_worker_recovery)
        
        # Setup event subscribers
        self.execution_monitor.add_subscriber(self._handle_test_event)
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                # Send initial status
                await self._send_status_update(websocket)
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client messages if needed
                    pass
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.discard(websocket)
                logger.info(f"Client disconnected: {websocket.remote_address}")
        
        self.websocket_server = await websockets.serve(
            handle_client, "localhost", self.websocket_port
        )
        logger.info(f"WebSocket server started on port {self.websocket_port}")
    
    async def _send_status_update(self, websocket=None):
        """Send status update to clients"""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "workers": [asdict(worker) for worker in self.worker_tracker.get_all_workers_health()],
            "active_tests": {
                name: asdict(event) for name, event in self.execution_monitor.get_active_tests().items()
            },
            "recent_events": [
                asdict(event) for event in self.execution_monitor.get_recent_events(10)
            ],
            "system_metrics": self._get_system_metrics()
        }
        
        message = json.dumps(status_data, default=str)
        
        if websocket:
            await websocket.send(message)
        else:
            # Broadcast to all connected clients
            disconnected_clients = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        workers = self.worker_tracker.get_all_workers_health()
        active_tests = self.execution_monitor.get_active_tests()
        
        healthy_workers = sum(1 for w in workers if w.is_healthy())
        total_tests_completed = sum(w.tests_executed for w in workers)
        total_tests_failed = sum(w.tests_failed for w in workers)
        
        # Calculate average test duration
        all_durations = []
        for test_name in self.execution_monitor.test_metrics:
            all_durations.extend(self.execution_monitor.test_metrics[test_name])
        avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
        
        # Get system resource usage
        system_cpu = psutil.cpu_percent()
        system_memory = psutil.virtual_memory().percent
        
        # Calculate overall performance score
        worker_scores = [w.performance_score for w in workers if w.is_healthy()]
        overall_score = sum(worker_scores) / len(worker_scores) if worker_scores else 0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            total_workers=len(workers),
            active_workers=len([w for w in workers if w.status == WorkerStatus.RUNNING]),
            healthy_workers=healthy_workers,
            total_tests_running=len(active_tests),
            tests_completed=total_tests_completed,
            tests_failed=total_tests_failed,
            avg_test_duration=avg_duration,
            system_cpu_usage=system_cpu,
            system_memory_usage=system_memory,
            overall_performance_score=overall_score
        )
    
    def _handle_worker_recovery(self, worker_id: str, worker: WorkerHealth):
        """Handle worker recovery"""
        logger.warning(f"Worker {worker_id} requires recovery (attempts: {worker.recovery_attempts})")
        
        # Implement recovery strategies
        if worker.recovery_attempts <= 3:
            # Try soft recovery first
            self._soft_recover_worker(worker_id, worker)
        else:
            # Hard recovery (restart worker)
            self._hard_recover_worker(worker_id, worker)
    
    def _soft_recover_worker(self, worker_id: str, worker: WorkerHealth):
        """Attempt soft recovery of worker"""
        try:
            # Send signal to worker process to reduce load
            if worker.status == WorkerStatus.OVERLOADED:
                # Could implement custom signal handling in workers
                pass
            
            logger.info(f"Attempted soft recovery for worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Soft recovery failed for worker {worker_id}: {e}")
    
    def _hard_recover_worker(self, worker_id: str, worker: WorkerHealth):
        """Attempt hard recovery (restart) of worker"""
        try:
            # Terminate the worker process
            try:
                process = psutil.Process(worker.process_id)
                process.terminate()
                process.wait(timeout=5)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.error(f'Error occurred: {e}')
            
            # Mark worker as terminated
            worker.status = WorkerStatus.TERMINATED
            
            logger.warning(f"Hard recovery initiated for worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Hard recovery failed for worker {worker_id}: {e}")
    
    def _handle_test_event(self, event: TestExecutionEvent):
        """Handle test execution event"""
        if event.event_type == "failed":
            logger.warning(f"Test failed: {event.test_name} on worker {event.worker_id}")
            
            # Update worker failure count
            worker = self.worker_tracker.get_worker_health(event.worker_id)
            if worker:
                worker.tests_failed += 1
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start WebSocket server
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_websocket_server())
            loop.run_forever()
        
        self.monitoring_thread = threading.Thread(target=run_websocket, daemon=True)
        self.monitoring_thread.start()
        
        # Start metrics collection
        self._start_metrics_collection()
        
        logger.info("Real-time monitoring system started")
    
    def _start_metrics_collection(self):
        """Start collecting system metrics"""
        def collect_metrics():
            while self.monitoring_active:
                try:
                    metrics = self._get_system_metrics()
                    self.metrics_history.append(asdict(metrics))
                    
                    # Broadcast updates to WebSocket clients
                    if self.connected_clients:
                        asyncio.run_coroutine_threadsafe(
                            self._send_status_update(),
                            asyncio.get_event_loop()
                        )
                    
                    time.sleep(1)  # Collect metrics every second
                    
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    time.sleep(5)
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        
        if self.websocket_server:
            self.websocket_server.close()
        
        logger.info("Real-time monitoring system stopped")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(self._get_system_metrics()),
            "worker_health": [asdict(w) for w in self.worker_tracker.get_all_workers_health()],
            "active_tests": {
                name: asdict(event) for name, event in self.execution_monitor.get_active_tests().items()
            },
            "recent_events": [
                asdict(event) for event in self.execution_monitor.get_recent_events(50)
            ],
            "metrics_history": list(self.metrics_history)[-100:],  # Last 100 data points
            "test_statistics": {
                test_name: self.execution_monitor.get_test_statistics(test_name)
                for test_name in list(self.execution_monitor.test_metrics.keys())[:20]
            }
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        workers = self.worker_tracker.get_all_workers_health()
        system_metrics = self._get_system_metrics()
        
        # Worker health summary
        healthy_workers = [w for w in workers if w.is_healthy()]
        unhealthy_workers = [w for w in workers if not w.is_healthy()]
        
        # Performance analysis
        performance_scores = [w.performance_score for w in workers]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        # Test execution analysis
        total_tests = sum(w.tests_executed for w in workers)
        total_failures = sum(w.tests_failed for w in workers)
        overall_success_rate = ((total_tests - total_failures) / total_tests) if total_tests > 0 else 0
        
        # System resource analysis
        resource_alerts = []
        if system_metrics.system_cpu_usage > 90:
            resource_alerts.append("High CPU usage detected")
        if system_metrics.system_memory_usage > 90:
            resource_alerts.append("High memory usage detected")
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_workers": len(workers),
                "healthy_workers": len(healthy_workers),
                "unhealthy_workers": len(unhealthy_workers),
                "avg_performance_score": avg_performance,
                "overall_success_rate": overall_success_rate,
                "total_tests_executed": total_tests,
                "active_tests": system_metrics.total_tests_running
            },
            "worker_details": {
                "healthy": [asdict(w) for w in healthy_workers],
                "unhealthy": [asdict(w) for w in unhealthy_workers]
            },
            "system_metrics": asdict(system_metrics),
            "resource_alerts": resource_alerts,
            "recommendations": self._generate_recommendations(workers, system_metrics)
        }
    
    def _generate_recommendations(self, workers: List[WorkerHealth], 
                                system_metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Worker-based recommendations
        overloaded_workers = [w for w in workers if w.status == WorkerStatus.OVERLOADED]
        if overloaded_workers:
            recommendations.append({
                "type": "worker_optimization",
                "severity": "high",
                "message": f"{len(overloaded_workers)} workers are overloaded",
                "action": "Reduce test load or add more workers"
            })
        
        # Performance-based recommendations
        low_performance_workers = [w for w in workers if w.performance_score < 60]
        if low_performance_workers:
            recommendations.append({
                "type": "performance",
                "severity": "medium",
                "message": f"{len(low_performance_workers)} workers have low performance",
                "action": "Investigate worker performance issues"
            })
        
        # System resource recommendations
        if system_metrics.system_cpu_usage > 85:
            recommendations.append({
                "type": "system_resources",
                "severity": "high",
                "message": "High system CPU usage",
                "action": "Reduce parallel workers or optimize test execution"
            })
        
        return recommendations


# Integration with pytest-xdist
class PytestXDistMonitor:
    """Monitor pytest-xdist workers"""
    
    def __init__(self, monitoring_system: RealTimeMonitoringSystem):
        self.monitoring_system = monitoring_system
        
    def pytest_configure_node(self, node):
        """Configure pytest worker node"""
        worker_id = getattr(node, 'workerid', 'master')
        
        if worker_id != 'master':
            # Register worker with monitoring system
            self.monitoring_system.worker_tracker.register_worker(worker_id, os.getpid())
    
    def pytest_runtest_setup(self, item):
        """Test setup hook"""
        worker_id = getattr(item.config, 'workerid', 'master')
        
        if worker_id != 'master':
            event = TestExecutionEvent(
                event_type="started",
                test_name=item.nodeid,
                worker_id=worker_id,
                timestamp=datetime.now()
            )
            self.monitoring_system.execution_monitor.record_event(event)
    
    def pytest_runtest_teardown(self, item, nextitem):
        """Test teardown hook"""
        worker_id = getattr(item.config, 'workerid', 'master')
        
        if worker_id != 'master':
            # Update worker health metrics
            self.monitoring_system.worker_tracker.update_worker_health(worker_id, {
                'tests_executed': 1,
                'response_time': 0.1  # Placeholder
            })


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    # Create monitoring system
    monitoring = RealTimeMonitoringSystem()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Simulate some workers and tests
    monitoring.worker_tracker.register_worker("worker_1", 12345)
    monitoring.worker_tracker.register_worker("worker_2", 12346)
    
    # Simulate test events
    for i in range(5):
        event = TestExecutionEvent(
            event_type="started",
            test_name=f"test_example_{i}",
            worker_id="worker_1",
            timestamp=datetime.now()
        )
        monitoring.execution_monitor.record_event(event)
        
        time.sleep(0.1)
        
        event = TestExecutionEvent(
            event_type="completed",
            test_name=f"test_example_{i}",
            worker_id="worker_1",
            timestamp=datetime.now(),
            duration=0.5
        )
        monitoring.execution_monitor.record_event(event)
    
    # Update worker health
    monitoring.worker_tracker.update_worker_health("worker_1", {
        'cpu_usage': 45.0,
        'memory_usage': 512.0,
        'tests_executed': 5,
        'tests_passed': 5,
        'tests_failed': 0,
        'response_time': 0.3
    })
    
    # Generate health report
    report = monitoring.generate_health_report()
    print(f"Health report: {json.dumps(report, indent=2, default=str)}")
    
    # Keep running for a bit
    time.sleep(5)
    
    # Stop monitoring
    monitoring.stop_monitoring()