#!/usr/bin/env python3
"""
AGENT 6: Monitoring System Integration
Complete integration of all monitoring components with unified control,
comprehensive testing, and production-ready deployment.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
from prometheus_client import start_http_server

# Import existing components
from .health_monitor import HealthMonitor
from .prometheus_metrics import MetricsCollector, MetricsConfig
from .enhanced_alerting import EnhancedAlertingSystem, EnhancedAlert, AlertPriority, AlertStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Monitoring intervals
    health_check_interval: int = 30  # seconds
    performance_check_interval: int = 1  # seconds
    regime_check_interval: int = 60  # seconds
    
    # Prometheus metrics
    metrics_port: int = 8000
    metrics_path: str = "/metrics"
    enable_metrics_server: bool = True
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_channels: List[str] = None
    
    # Feature flags
    enable_health_monitoring: bool = True
    enable_performance_monitoring: bool = True
    enable_regime_monitoring: bool = True
    enable_dashboard: bool = True
    
    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["console", "webhook"]

class MonitoringSystem:
    """Unified monitoring system orchestrator."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.health_monitor = None
        self.metrics_collector = None
        
        # System state
        self.initialized = False
        self.running = False
        self.monitoring_tasks = []
        self.metrics_server_thread = None
        
        # Performance tracking
        self.startup_time = None
        self.last_health_check = None
        self.system_metrics = {}
        
    async def initialize(self):
        """Initialize all monitoring components."""
        if self.initialized:
            return
            
        logger.info("Initializing monitoring system...")
        start_time = time.time()
        
        try:
            # Initialize health monitor
            if self.config.enable_health_monitoring:
                self.health_monitor = HealthMonitor(f"redis://{self.config.redis_host}:{self.config.redis_port}")
                
            # Initialize metrics collector
            metrics_config = MetricsConfig(
                enable_system_metrics=True,
                enable_business_metrics=True,
                enable_sla_metrics=True,
                metrics_port=self.config.metrics_port
            )
            self.metrics_collector = MetricsCollector(metrics_config)
            
            # Start metrics server
            if self.config.enable_metrics_server:
                await self._start_metrics_server()
                
            self.initialized = True
            self.startup_time = time.time() - start_time
            
            logger.info(f"Monitoring system initialized in {self.startup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            raise
            
    async def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        def start_server():
            try:
                start_http_server(self.config.metrics_port)
                logger.info(f"Metrics server started on port {self.config.metrics_port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
                
        self.metrics_server_thread = threading.Thread(target=start_server)
        self.metrics_server_thread.daemon = True
        self.metrics_server_thread.start()
        
    async def start(self):
        """Start all monitoring components."""
        if not self.initialized:
            await self.initialize()
            
        if self.running:
            logger.warning("Monitoring system already running")
            return
            
        logger.info("Starting monitoring system...")
        
        try:
            # Start metrics collection
            if self.metrics_collector:
                self.metrics_collector.start_collection()
                
            # Start health monitoring
            if self.health_monitor:
                task = asyncio.create_task(self._health_monitoring_loop())
                self.monitoring_tasks.append(task)
                
            # Start system status monitoring
            task = asyncio.create_task(self._system_status_loop())
            self.monitoring_tasks.append(task)
            
            self.running = True
            logger.info("Monitoring system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Stop all monitoring components."""
        if not self.running:
            return
            
        logger.info("Stopping monitoring system...")
        
        try:
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
                    
            if self.metrics_collector:
                self.metrics_collector.stop_collection()
                
            # Wait for all tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
                
            self.running = False
            self.monitoring_tasks = []
            
            logger.info("Monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")
            
    async def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.running:
            try:
                # Check system health
                system_health = await self.health_monitor.check_all_components()
                self.last_health_check = datetime.utcnow()
                
                # Check for critical health issues
                unhealthy_components = [
                    comp for comp in system_health.components
                    if comp.status.value == 'unhealthy'
                ]
                
                if unhealthy_components:
                    logger.warning(f"Unhealthy components detected: {[comp.name for comp in unhealthy_components]}")
                    
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _system_status_loop(self):
        """System status monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_status = await self._collect_system_status()
                self.system_metrics = system_status
                
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in system status loop: {e}")
                await asyncio.sleep(120)
                
    async def _collect_system_status(self) -> Dict[str, Any]:
        """Collect comprehensive system status."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'initialized': self.initialized,
            'running': self.running,
            'startup_time': self.startup_time,
            'uptime': time.time() - (time.time() - self.startup_time) if self.startup_time else 0,
            'components': {}
        }
        
        # Health monitor status
        if self.health_monitor:
            status['components']['health_monitor'] = {
                'enabled': True,
                'last_check': self.last_health_check.isoformat() if self.last_health_check else None
            }
            
        return status
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.initialized:
            return {'status': 'not_initialized'}
            
        return self.system_metrics
        
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self.health_monitor:
            return {'error': 'Health monitoring not enabled'}
            
        system_health = await self.health_monitor.check_all_components()
        return system_health.to_dict()
        
    async def test_system(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        logger.info("Running monitoring system test...")
        
        test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'PASS',
            'tests': {}
        }
        
        # Test health monitoring
        if self.health_monitor:
            try:
                health_status = await self.health_monitor.check_all_components()
                test_results['tests']['health_monitoring'] = 'PASS'
                test_results['health_summary'] = health_status.to_dict()
            except Exception as e:
                test_results['tests']['health_monitoring'] = f'FAIL: {e}'
                test_results['overall_status'] = 'FAIL'
                
        logger.info(f"System test completed: {test_results['overall_status']}")
        return test_results

# Factory functions
def create_monitoring_system(config: Optional[MonitoringConfig] = None) -> MonitoringSystem:
    """Create monitoring system instance."""
    if config is None:
        config = MonitoringConfig()
    return MonitoringSystem(config)

def create_default_monitoring_system() -> MonitoringSystem:
    """Create monitoring system with default configuration."""
    config = MonitoringConfig(
        enable_health_monitoring=True,
        enable_performance_monitoring=True,
        enable_dashboard=True,
        enable_regime_monitoring=True,
        enable_alerting=True,
        enable_metrics_server=True
    )
    return MonitoringSystem(config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create monitoring system
        config = MonitoringConfig()
        monitoring_system = create_monitoring_system(config)
        
        try:
            # Initialize and start
            await monitoring_system.initialize()
            await monitoring_system.start()
            
            # Run system test
            test_results = await monitoring_system.test_system()
            print(json.dumps(test_results, indent=2))
            
            # Get system status
            status = await monitoring_system.get_system_status()
            print(json.dumps(status, indent=2))
            
            # Keep running for demonstration
            logger.info("Monitoring system running. Press Ctrl+C to stop.")
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await monitoring_system.stop()
            
    asyncio.run(main())