#!/usr/bin/env python3
"""
Real-time RTO Monitoring and Measurement System
Continuously monitors database availability and measures RTO during failovers
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import psycopg2
import requests
from dataclasses import dataclass, asdict
import threading
from collections import deque
import statistics
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AvailabilityEvent:
    """Database availability event"""
    timestamp: datetime
    event_type: str  # 'up', 'down', 'recovery'
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    primary_node: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'response_time_ms': self.response_time_ms,
            'error_message': self.error_message,
            'primary_node': self.primary_node
        }

@dataclass
class RTOMetrics:
    """RTO measurement metrics"""
    downtime_start: datetime
    downtime_end: datetime
    rto_seconds: float
    recovery_type: str  # 'automatic', 'manual'
    affected_connections: int
    primary_before: str
    primary_after: str
    
    def to_dict(self) -> Dict:
        return {
            'downtime_start': self.downtime_start.isoformat(),
            'downtime_end': self.downtime_end.isoformat(),
            'rto_seconds': self.rto_seconds,
            'recovery_type': self.recovery_type,
            'affected_connections': self.affected_connections,
            'primary_before': self.primary_before,
            'primary_after': self.primary_after
        }

class DatabaseRTOMonitor:
    """Real-time RTO monitoring system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.postgres_config = config.get('postgres', {})
        self.patroni_api_url = config.get('patroni_api_url', 'http://localhost:8008')
        self.check_interval = config.get('check_interval', 1)  # seconds
        self.monitoring = False
        self.current_state = 'unknown'
        self.last_primary = None
        self.downtime_start = None
        
        # Event storage
        self.events = deque(maxlen=10000)  # Store last 10k events
        self.rto_measurements = deque(maxlen=1000)  # Store last 1k RTO measurements
        
        # Statistics
        self.uptime_percentage = 0.0
        self.average_response_time = 0.0
        self.total_checks = 0
        self.successful_checks = 0
        
        # Alerts
        self.alert_threshold_seconds = config.get('alert_threshold_seconds', 30)
        self.alert_callbacks = []
        
    async def get_cluster_info(self) -> Dict:
        """Get cluster information from Patroni"""
        try:
            response = requests.get(f"{self.patroni_api_url}/cluster", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Failed to get cluster info: {e}")
            return {}
    
    async def get_primary_node(self) -> Optional[str]:
        """Get current primary node"""
        cluster_info = await self.get_cluster_info()
        for member in cluster_info.get('members', []):
            if member.get('role') == 'Leader':
                return member.get('name')
        return None
    
    async def check_database_connectivity(self) -> Tuple[bool, Optional[float], Optional[str]]:
        """Check database connectivity and measure response time"""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=self.postgres_config.get('host', 'localhost'),
                port=self.postgres_config.get('port', 5432),
                database=self.postgres_config.get('database', 'grandmodel'),
                user=self.postgres_config.get('user', 'grandmodel'),
                password=self.postgres_config.get('password'),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT 1, current_timestamp')
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            return True, response_time, None
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, response_time, str(e)
    
    async def record_event(self, event_type: str, response_time_ms: Optional[float] = None, 
                          error_message: Optional[str] = None):
        """Record an availability event"""
        primary_node = await self.get_primary_node()
        
        event = AvailabilityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            response_time_ms=response_time_ms,
            error_message=error_message,
            primary_node=primary_node
        )
        
        self.events.append(event)
        
        # Log important events
        if event_type in ['down', 'recovery']:
            logger.info(f"Database {event_type} event recorded. Primary: {primary_node}")
    
    async def detect_failover(self) -> bool:
        """Detect if a failover has occurred"""
        current_primary = await self.get_primary_node()
        
        if self.last_primary and current_primary and self.last_primary != current_primary:
            logger.info(f"Failover detected: {self.last_primary} -> {current_primary}")
            return True
        
        return False
    
    async def measure_rto(self, downtime_start: datetime, downtime_end: datetime,
                         primary_before: str, primary_after: str) -> RTOMetrics:
        """Measure RTO for a downtime period"""
        rto_seconds = (downtime_end - downtime_start).total_seconds()
        
        # Determine recovery type based on failover detection
        recovery_type = 'automatic' if await self.detect_failover() else 'manual'
        
        # Estimate affected connections (simplified)
        affected_connections = self.estimate_affected_connections()
        
        metrics = RTOMetrics(
            downtime_start=downtime_start,
            downtime_end=downtime_end,
            rto_seconds=rto_seconds,
            recovery_type=recovery_type,
            affected_connections=affected_connections,
            primary_before=primary_before,
            primary_after=primary_after
        )
        
        self.rto_measurements.append(metrics)
        
        # Alert if RTO exceeds threshold
        if rto_seconds > self.alert_threshold_seconds:
            await self.trigger_rto_alert(metrics)
        
        return metrics
    
    def estimate_affected_connections(self) -> int:
        """Estimate number of affected connections during downtime"""
        # This is a simplified estimation
        # In a real implementation, you'd query active connections
        return 10  # Placeholder
    
    async def trigger_rto_alert(self, metrics: RTOMetrics):
        """Trigger RTO alert when threshold is exceeded"""
        alert_message = f"RTO Alert: {metrics.rto_seconds:.2f}s exceeds threshold of {self.alert_threshold_seconds}s"
        logger.warning(alert_message)
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting RTO monitoring...")
        
        while self.monitoring:
            try:
                # Check database connectivity
                is_up, response_time, error = await self.check_database_connectivity()
                current_primary = await self.get_primary_node()
                
                # Update statistics
                self.total_checks += 1
                if is_up:
                    self.successful_checks += 1
                
                # State transition logic
                if self.current_state == 'unknown':
                    if is_up:
                        self.current_state = 'up'
                        self.last_primary = current_primary
                        await self.record_event('up', response_time)
                    else:
                        self.current_state = 'down'
                        self.downtime_start = datetime.now()
                        await self.record_event('down', response_time, error)
                
                elif self.current_state == 'up':
                    if not is_up:
                        # Database went down
                        self.current_state = 'down'
                        self.downtime_start = datetime.now()
                        await self.record_event('down', response_time, error)
                        logger.warning(f"Database down detected at {self.downtime_start}")
                    else:
                        # Check for failover
                        if await self.detect_failover():
                            # Failover without downtime
                            old_primary = self.last_primary
                            self.last_primary = current_primary
                            await self.record_event('failover', response_time)
                            logger.info(f"Seamless failover: {old_primary} -> {current_primary}")
                        else:
                            # Normal operation
                            await self.record_event('up', response_time)
                
                elif self.current_state == 'down':
                    if is_up:
                        # Database recovered
                        self.current_state = 'up'
                        downtime_end = datetime.now()
                        
                        # Measure RTO
                        if self.downtime_start:
                            rto_metrics = await self.measure_rto(
                                self.downtime_start, 
                                downtime_end,
                                self.last_primary or 'unknown',
                                current_primary or 'unknown'
                            )
                            
                            logger.info(f"Recovery detected. RTO: {rto_metrics.rto_seconds:.2f}s")
                        
                        self.last_primary = current_primary
                        await self.record_event('recovery', response_time)
                        self.downtime_start = None
                    else:
                        # Still down
                        await self.record_event('down', response_time, error)
                
                # Update running statistics
                self.uptime_percentage = (self.successful_checks / self.total_checks) * 100
                
                # Calculate average response time for successful checks
                recent_successful_events = [
                    e for e in list(self.events)[-100:]  # Last 100 events
                    if e.event_type == 'up' and e.response_time_ms is not None
                ]
                
                if recent_successful_events:
                    self.average_response_time = statistics.mean([
                        e.response_time_ms for e in recent_successful_events
                    ])
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring = True
        await self.monitoring_loop()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring = False
        logger.info("RTO monitoring stopped")
    
    def get_current_stats(self) -> Dict:
        """Get current monitoring statistics"""
        recent_rto_measurements = list(self.rto_measurements)[-10:]  # Last 10 measurements
        
        stats = {
            'monitoring_status': 'active' if self.monitoring else 'stopped',
            'current_state': self.current_state,
            'current_primary': self.last_primary,
            'uptime_percentage': self.uptime_percentage,
            'average_response_time_ms': self.average_response_time,
            'total_checks': self.total_checks,
            'successful_checks': self.successful_checks,
            'total_events': len(self.events),
            'total_rto_measurements': len(self.rto_measurements)
        }
        
        if recent_rto_measurements:
            rto_times = [m.rto_seconds for m in recent_rto_measurements]
            stats['rto_statistics'] = {
                'recent_average_rto': statistics.mean(rto_times),
                'recent_min_rto': min(rto_times),
                'recent_max_rto': max(rto_times),
                'measurements_count': len(recent_rto_measurements),
                'target_achievement_rate': sum(1 for rto in rto_times if rto < 30) / len(rto_times) * 100
            }
        
        return stats
    
    def get_recent_events(self, count: int = 100) -> List[Dict]:
        """Get recent events"""
        return [event.to_dict() for event in list(self.events)[-count:]]
    
    def get_rto_measurements(self, count: int = 100) -> List[Dict]:
        """Get recent RTO measurements"""
        return [measurement.to_dict() for measurement in list(self.rto_measurements)[-count:]]
    
    def export_metrics(self, filename: str):
        """Export metrics to file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_current_stats(),
            'recent_events': self.get_recent_events(),
            'rto_measurements': self.get_rto_measurements()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filename}")

class RTODashboard:
    """Simple web dashboard for RTO monitoring"""
    
    def __init__(self, monitor: DatabaseRTOMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
    
    async def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        stats = self.monitor.get_current_stats()
        recent_events = self.monitor.get_recent_events(10)
        rto_measurements = self.monitor.get_rto_measurements(10)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Database RTO Monitor</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .status-up {{ color: green; }}
                .status-down {{ color: red; }}
                .status-unknown {{ color: orange; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
                .alert {{ background: #ffcccc; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Database RTO Monitor Dashboard</h1>
            
            <div class="metric">
                <h2>Current Status</h2>
                <p>State: <span class="status-{stats['current_state']}">{stats['current_state'].upper()}</span></p>
                <p>Primary Node: {stats['current_primary'] or 'Unknown'}</p>
                <p>Uptime: {stats['uptime_percentage']:.2f}%</p>
                <p>Average Response Time: {stats['average_response_time_ms']:.2f}ms</p>
            </div>
            
            {self._generate_rto_section(stats)}
            
            <div class="metric">
                <h2>Recent Events</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Event Type</th>
                        <th>Response Time (ms)</th>
                        <th>Primary Node</th>
                    </tr>
                    {self._generate_events_table(recent_events)}
                </table>
            </div>
            
            <div class="metric">
                <h2>RTO Measurements</h2>
                <table>
                    <tr>
                        <th>Downtime Start</th>
                        <th>RTO (seconds)</th>
                        <th>Recovery Type</th>
                        <th>Primary Before</th>
                        <th>Primary After</th>
                    </tr>
                    {self._generate_rto_table(rto_measurements)}
                </table>
            </div>
            
            <p><em>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        return html
    
    def _generate_rto_section(self, stats: Dict) -> str:
        """Generate RTO statistics section"""
        if 'rto_statistics' not in stats:
            return '<div class="metric"><h2>RTO Statistics</h2><p>No RTO measurements available</p></div>'
        
        rto_stats = stats['rto_statistics']
        alert_class = "alert" if rto_stats['recent_average_rto'] > 30 else "metric"
        
        return f"""
        <div class="{alert_class}">
            <h2>RTO Statistics</h2>
            <p>Recent Average RTO: {rto_stats['recent_average_rto']:.2f}s</p>
            <p>Recent Min RTO: {rto_stats['recent_min_rto']:.2f}s</p>
            <p>Recent Max RTO: {rto_stats['recent_max_rto']:.2f}s</p>
            <p>Target Achievement Rate: {rto_stats['target_achievement_rate']:.1f}%</p>
            <p>Measurements: {rto_stats['measurements_count']}</p>
        </div>
        """
    
    def _generate_events_table(self, events: List[Dict]) -> str:
        """Generate events table rows"""
        rows = []
        for event in reversed(events):  # Show most recent first
            rows.append(f"""
                <tr>
                    <td>{event['timestamp']}</td>
                    <td>{event['event_type']}</td>
                    <td>{event['response_time_ms']:.2f if event['response_time_ms'] else 'N/A'}</td>
                    <td>{event['primary_node'] or 'Unknown'}</td>
                </tr>
            """)
        return ''.join(rows)
    
    def _generate_rto_table(self, measurements: List[Dict]) -> str:
        """Generate RTO measurements table rows"""
        rows = []
        for measurement in reversed(measurements):  # Show most recent first
            rto_class = "alert" if measurement['rto_seconds'] > 30 else ""
            rows.append(f"""
                <tr class="{rto_class}">
                    <td>{measurement['downtime_start']}</td>
                    <td>{measurement['rto_seconds']:.2f}</td>
                    <td>{measurement['recovery_type']}</td>
                    <td>{measurement['primary_before']}</td>
                    <td>{measurement['primary_after']}</td>
                </tr>
            """)
        return ''.join(rows)

# Configuration
DEFAULT_CONFIG = {
    'patroni_api_url': 'http://localhost:8008',
    'postgres': {
        'host': 'localhost',
        'port': 5432,
        'database': 'grandmodel',
        'user': 'grandmodel',
        'password': 'your_password_here'
    },
    'check_interval': 1,
    'alert_threshold_seconds': 30
}

async def main():
    """Main monitoring application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database RTO Monitor')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--export', help='Export metrics to file')
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create monitor
    monitor = DatabaseRTOMonitor(config)
    
    # Set up signal handlers for graceful shutdown
    import signal
    def signal_handler(signum, frame):
        logger.info("Shutting down...")
        monitor.stop_monitoring()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    try:
        if args.dashboard:
            # Start dashboard in separate thread
            dashboard = RTODashboard(monitor, args.port)
            # Dashboard implementation would go here
            logger.info(f"Dashboard would be available at http://localhost:{args.port}")
        
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        if args.export:
            monitor.export_metrics(args.export)

if __name__ == "__main__":
    asyncio.run(main())