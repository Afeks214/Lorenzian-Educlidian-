"""
Comprehensive RTO Monitoring System Integration Module.

This module provides a unified interface for all RTO monitoring components:
- RTO monitoring and measurement
- Real-time dashboard
- Alerting system
- Historical analytics
- Validation testing
- System coordination
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from src.monitoring.rto_monitor import RTOMonitoringSystem, initialize_rto_monitor
from src.monitoring.rto_dashboard import RTODashboard, create_rto_dashboard
from src.monitoring.rto_alerting import RTOAlertingSystem, initialize_alerting_system
from src.monitoring.rto_analytics import RTOAnalyticsSystem, initialize_analytics_system
from src.monitoring.rto_validation import RTOValidationFramework, initialize_validation_framework
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class RTOSystemConfig:
    """RTO system configuration."""
    # Database configuration
    database_config: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading',
        'user': 'postgres',
        'password': 'password'
    })
    
    # Trading engine configuration
    trading_engine_config: Dict[str, Any] = field(default_factory=lambda: {
        'health_endpoint': 'http://localhost:8000/health'
    })
    
    # Alerting configuration
    alerting_config: Dict[str, Any] = field(default_factory=lambda: {
        "email": {
            "enabled": False,
            "host": "smtp.gmail.com",
            "port": 587,
            "use_tls": True,
            "username": "",
            "password": "",
            "from": "rto-alerts@trading-system.com",
            "recipients": ["ops@trading-system.com"]
        },
        "slack": {
            "enabled": False,
            "webhook_url": "",
            "channels": ["#alerts"]
        },
        "webhook": {
            "enabled": False,
            "endpoints": ["https://your-webhook-endpoint.com/alerts"]
        }
    })
    
    # Dashboard configuration
    dashboard_config: Dict[str, Any] = field(default_factory=lambda: {
        'host': '0.0.0.0',
        'port': 8001
    })
    
    # Monitoring configuration
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        'check_interval': 10.0,
        'auto_start': True
    })
    
    # Validation configuration
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_continuous_testing': False,
        'smoke_test_interval': 3600,  # 1 hour
        'full_validation_interval': 86400  # 24 hours
    })

class RTOSystem:
    """Comprehensive RTO monitoring system."""
    
    def __init__(self, config: RTOSystemConfig):
        self.config = config
        self.event_bus = EventBus()
        
        # Initialize components
        self.rto_monitor = None
        self.dashboard = None
        self.alerting_system = None
        self.analytics_system = None
        self.validation_framework = None
        
        # System state
        self._running = False
        self._tasks = []
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RTO system components."""
        logger.info("Initializing RTO monitoring system components")
        
        # Initialize RTO monitor
        self.rto_monitor = initialize_rto_monitor(
            self.config.database_config,
            self.config.trading_engine_config
        )
        
        # Initialize alerting system
        self.alerting_system = initialize_alerting_system(self.config.alerting_config)
        
        # Initialize analytics system
        self.analytics_system = initialize_analytics_system()
        
        # Initialize validation framework
        self.validation_framework = initialize_validation_framework(
            self.rto_monitor,
            self.analytics_system
        )
        
        # Initialize dashboard
        self.dashboard = RTODashboard(self.rto_monitor)
        
        logger.info("RTO monitoring system components initialized successfully")
    
    async def start(self):
        """Start the RTO monitoring system."""
        if self._running:
            logger.warning("RTO system is already running")
            return
        
        logger.info("Starting RTO monitoring system")
        self._running = True
        
        try:
            # Start RTO monitoring
            if self.config.monitoring_config.get('auto_start', True):
                await self.rto_monitor.start_monitoring(
                    self.config.monitoring_config.get('check_interval', 10.0)
                )
            
            # Start continuous validation if enabled
            if self.config.validation_config.get('enable_continuous_testing', False):
                self._tasks.append(asyncio.create_task(self._continuous_validation_loop()))
            
            # Start dashboard
            dashboard_task = asyncio.create_task(
                self.dashboard.start(
                    self.config.dashboard_config.get('host', '0.0.0.0'),
                    self.config.dashboard_config.get('port', 8001)
                )
            )
            self._tasks.append(dashboard_task)
            
            logger.info("RTO monitoring system started successfully")
            
            # Keep the main coroutine running
            await asyncio.gather(*self._tasks)
            
        except Exception as e:
            logger.error(f"Error starting RTO system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the RTO monitoring system."""
        if not self._running:
            return
        
        logger.info("Stopping RTO monitoring system")
        self._running = False
        
        # Stop monitoring
        if self.rto_monitor:
            await self.rto_monitor.stop_monitoring()
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("RTO monitoring system stopped")
    
    async def _continuous_validation_loop(self):
        """Continuous validation loop."""
        logger.info("Starting continuous validation loop")
        
        smoke_test_interval = self.config.validation_config.get('smoke_test_interval', 3600)
        full_validation_interval = self.config.validation_config.get('full_validation_interval', 86400)
        
        last_smoke_test = 0
        last_full_validation = 0
        
        while self._running:
            try:
                current_time = datetime.utcnow().timestamp()
                
                # Run smoke tests
                if current_time - last_smoke_test >= smoke_test_interval:
                    logger.info("Running scheduled smoke tests")
                    await self.validation_framework.run_smoke_tests()
                    last_smoke_test = current_time
                
                # Run full validation
                if current_time - last_full_validation >= full_validation_interval:
                    logger.info("Running scheduled full validation")
                    await self.validation_framework.run_full_validation()
                    last_full_validation = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in continuous validation loop: {e}")
                await asyncio.sleep(60)
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        logger.info("Running comprehensive health check")
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_running": self._running,
            "components": {},
            "overall_health": "healthy"
        }
        
        try:
            # Check RTO monitor
            if self.rto_monitor:
                rto_summary = self.rto_monitor.get_rto_summary(1)
                health_status["components"]["rto_monitor"] = {
                    "status": "healthy",
                    "summary": rto_summary
                }
            
            # Check alerting system
            if self.alerting_system:
                alert_summary = self.alerting_system.get_alert_summary(1)
                health_status["components"]["alerting_system"] = {
                    "status": "healthy",
                    "summary": alert_summary
                }
            
            # Check analytics system
            if self.analytics_system:
                health_status["components"]["analytics_system"] = {
                    "status": "healthy",
                    "cache_size": len(self.analytics_system._cache)
                }
            
            # Check validation framework
            if self.validation_framework:
                validation_status = self.validation_framework.get_validation_status()
                health_status["components"]["validation_framework"] = {
                    "status": "healthy",
                    "validation_status": validation_status
                }
            
            # Determine overall health
            component_statuses = [c.get("status", "unknown") for c in health_status["components"].values()]
            if all(status == "healthy" for status in component_statuses):
                health_status["overall_health"] = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                health_status["overall_health"] = "unhealthy"
            else:
                health_status["overall_health"] = "degraded"
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status["overall_health"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    async def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests."""
        return await self.validation_framework.run_smoke_tests()
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run full validation."""
        return await self.validation_framework.run_full_validation()
    
    async def run_load_tests(self, component: str, concurrent_failures: int = 3) -> Dict[str, Any]:
        """Run load tests."""
        return await self.validation_framework.run_load_tests(component, concurrent_failures)
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        return await self.validation_framework.generate_compliance_report()
    
    def get_comprehensive_analysis(self, component: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analysis."""
        return self.analytics_system.get_comprehensive_analysis(component, days)
    
    def get_comparative_analysis(self, components: List[str], days: int = 30) -> Dict[str, Any]:
        """Get comparative analysis."""
        return self.analytics_system.get_comparative_analysis(components, days)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self._running,
            "components": {
                "rto_monitor": self.rto_monitor is not None,
                "dashboard": self.dashboard is not None,
                "alerting_system": self.alerting_system is not None,
                "analytics_system": self.analytics_system is not None,
                "validation_framework": self.validation_framework is not None
            },
            "active_tasks": len(self._tasks),
            "config": {
                "monitoring_interval": self.config.monitoring_config.get('check_interval', 10.0),
                "dashboard_port": self.config.dashboard_config.get('port', 8001),
                "continuous_testing": self.config.validation_config.get('enable_continuous_testing', False)
            }
        }

# Global system instance
rto_system = None

def create_rto_system(config: RTOSystemConfig = None) -> RTOSystem:
    """Create RTO system instance."""
    global rto_system
    if config is None:
        config = RTOSystemConfig()
    rto_system = RTOSystem(config)
    return rto_system

def get_rto_system() -> Optional[RTOSystem]:
    """Get global RTO system instance."""
    return rto_system

# CLI interface for easy system management
async def main():
    """Main CLI interface."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="RTO Monitoring System")
    parser.add_argument("command", choices=["start", "health", "smoke", "validate", "load", "compliance", "status"], 
                       help="Command to execute")
    parser.add_argument("--component", help="Component name for specific operations")
    parser.add_argument("--days", type=int, default=30, help="Number of days for analysis")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = RTOSystemConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            # Update config with loaded data
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create system
    system = create_rto_system(config)
    
    try:
        if args.command == "start":
            print("Starting RTO monitoring system...")
            await system.start()
        
        elif args.command == "health":
            health = await system.run_health_check()
            print(json.dumps(health, indent=2))
        
        elif args.command == "smoke":
            results = await system.run_smoke_tests()
            print(json.dumps(results, indent=2))
        
        elif args.command == "validate":
            results = await system.run_full_validation()
            print(json.dumps(results, indent=2))
        
        elif args.command == "load":
            if not args.component:
                print("Error: --component required for load testing")
                sys.exit(1)
            results = await system.run_load_tests(args.component)
            print(json.dumps(results, indent=2))
        
        elif args.command == "compliance":
            report = await system.generate_compliance_report()
            print(json.dumps(report, indent=2))
        
        elif args.command == "status":
            status = system.get_system_status()
            print(json.dumps(status, indent=2))
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        await system.stop()
    except Exception as e:
        print(f"Error: {e}")
        await system.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())