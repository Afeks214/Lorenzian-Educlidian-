#!/usr/bin/env python3
"""
AGENT 13: Monitoring Integration and End-to-End Testing
Integrates all monitoring components with the trading system for comprehensive observability
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import redis
import os
from prometheus_client import start_http_server, CollectorRegistry, REGISTRY

# Import monitoring components
from .prometheus_metrics import (
    MetricsCollector, MetricsConfig, TradingMetricsCollector,
    RiskMetricsCollector, MARLMetricsCollector, SystemMetricsCollector,
    BusinessMetricsCollector, PerformanceTracker
)
from .sla_monitor import (
    SLAMonitor, SLATarget, SLAType, create_sla_monitor,
    DEFAULT_TRADING_SLA_TARGETS
)
from .enhanced_alerting import (
    EnhancedAlertingSystem, EnhancedAlert, AlertPriority, AlertStatus,
    LogComponent, AlertCorrelationRule, AlertSuppressionRule, EscalationPolicy
)
from .structured_logging import (
    StructuredLoggingSystem, LoggingConfiguration, StructuredLogger,
    correlation_context, LogContext, LogComponent as LogComp, setup_logging
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    # Prometheus configuration
    prometheus_port: int = 8000
    prometheus_enabled: bool = True
    
    # SLA monitoring configuration
    sla_monitoring_enabled: bool = True
    sla_check_interval: int = 30  # seconds
    
    # Alerting configuration
    alerting_enabled: bool = True
    alert_channels: List[str] = None
    
    # Logging configuration
    structured_logging_enabled: bool = True
    log_level: str = "INFO"
    log_file_path: str = "logs/grandmodel.log"
    elasticsearch_enabled: bool = False
    redis_enabled: bool = True
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Integration configuration
    health_check_interval: int = 60  # seconds
    performance_monitoring_enabled: bool = True
    business_metrics_enabled: bool = True
    
    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["console", "redis"]

class MonitoringSystem:
    """Comprehensive monitoring system that integrates all components."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.running = False
        self.monitoring_thread = None
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db
        )
        
        # Initialize components
        self.metrics_collector = None
        self.sla_monitor = None
        self.alerting_system = None
        self.logging_system = None
        self.structured_logger = None
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Health status
        self.component_health = {}
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # Initialize metrics collector
            if self.config.prometheus_enabled:
                metrics_config = MetricsConfig(
                    enable_system_metrics=True,
                    enable_business_metrics=self.config.business_metrics_enabled,
                    enable_sla_metrics=True,
                    collection_interval=1.0,
                    metrics_port=self.config.prometheus_port
                )
                self.metrics_collector = MetricsCollector(metrics_config)
                self.component_health['metrics_collector'] = True
                logger.info("Metrics collector initialized")
                
            # Initialize SLA monitor
            if self.config.sla_monitoring_enabled:
                self.sla_monitor = create_sla_monitor(
                    self.redis_client,
                    include_default_targets=True
                )
                self.component_health['sla_monitor'] = True
                logger.info("SLA monitor initialized")
                
            # Initialize alerting system
            if self.config.alerting_enabled:
                self.alerting_system = EnhancedAlertingSystem(self.redis_client)
                self.component_health['alerting_system'] = True
                logger.info("Alerting system initialized")
                
            # Initialize structured logging
            if self.config.structured_logging_enabled:
                logging_config = LoggingConfiguration()
                logging_config.console_enabled = True
                logging_config.file_enabled = True
                logging_config.file_path = self.config.log_file_path
                logging_config.elasticsearch_enabled = self.config.elasticsearch_enabled
                logging_config.redis_enabled = self.config.redis_enabled
                logging_config.redis_host = self.config.redis_host
                logging_config.redis_port = self.config.redis_port
                logging_config.redis_db = self.config.redis_db
                
                self.logging_system = StructuredLoggingSystem(logging_config)
                self.structured_logger = self.logging_system.get_logger("monitoring_system")
                self.component_health['logging_system'] = True
                logger.info("Structured logging initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}")
            raise
            
    def start(self):
        """Start the monitoring system."""
        if self.running:
            logger.warning("Monitoring system is already running")
            return
            
        self.running = True
        
        try:
            # Start metrics collector
            if self.metrics_collector:
                self.metrics_collector.start_collection()
                self.metrics_collector.start_metrics_server()
                
            # Start SLA monitor
            if self.sla_monitor:
                self.sla_monitor.start_monitoring()
                
            # Start health monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            if self.structured_logger:
                self.structured_logger.info(
                    "Monitoring system started successfully",
                    extra_fields={
                        'components': list(self.component_health.keys()),
                        'config': {
                            'prometheus_port': self.config.prometheus_port,
                            'sla_monitoring': self.config.sla_monitoring_enabled,
                            'alerting': self.config.alerting_enabled
                        }
                    }
                )
            else:
                logger.info("Monitoring system started successfully")
                
        except Exception as e:
            self.running = False
            logger.error(f"Failed to start monitoring system: {e}")
            raise
            
    def stop(self):
        """Stop the monitoring system."""
        if not self.running:
            return
            
        self.running = False
        
        try:
            # Stop metrics collector
            if self.metrics_collector:
                self.metrics_collector.stop_collection()
                
            # Stop SLA monitor
            if self.sla_monitor:
                self.sla_monitor.stop_monitoring()
                
            # Stop logging system
            if self.logging_system:
                self.logging_system.shutdown()
                
            # Wait for monitoring thread to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
                
            if self.structured_logger:
                self.structured_logger.info("Monitoring system stopped")
            else:
                logger.info("Monitoring system stopped")
                
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {e}")
            
    def _monitoring_loop(self):
        """Main monitoring loop for health checks and maintenance."""
        while self.running:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Update component health metrics
                self._update_health_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep until next cycle
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
                
    def _perform_health_checks(self):
        """Perform health checks on all components."""
        health_status = {}
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status['redis'] = True
        except Exception as e:
            health_status['redis'] = False
            logger.error(f"Redis health check failed: {e}")
            
        # Check metrics collector
        if self.metrics_collector:
            try:
                # Simple health check - try to get current metrics
                metrics = self.metrics_collector.get_metrics()
                health_status['metrics_collector'] = bool(metrics)
            except Exception as e:
                health_status['metrics_collector'] = False
                logger.error(f"Metrics collector health check failed: {e}")
                
        # Check SLA monitor
        if self.sla_monitor:
            try:
                # Check if SLA monitor is responsive
                status = self.sla_monitor.get_sla_status("trading_engine")
                health_status['sla_monitor'] = bool(status)
            except Exception as e:
                health_status['sla_monitor'] = False
                logger.error(f"SLA monitor health check failed: {e}")
                
        # Check alerting system
        if self.alerting_system:
            try:
                # Check if alerting system is responsive
                status = asyncio.run(self.alerting_system.get_alert_status())
                health_status['alerting_system'] = bool(status)
            except Exception as e:
                health_status['alerting_system'] = False
                logger.error(f"Alerting system health check failed: {e}")
                
        # Update component health
        self.component_health.update(health_status)
        
    def _update_health_metrics(self):
        """Update health metrics for monitoring components."""
        if self.metrics_collector:
            # Update system metrics
            self.metrics_collector.system_metrics.collect_system_metrics()
            
            # Update health status metrics
            for component, health in self.component_health.items():
                # This would be a custom gauge metric for component health
                pass
                
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            # Clean up old alerts from Redis
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # This is a simplified cleanup - in production, you'd want more sophisticated retention policies
            alert_keys = self.redis_client.keys("alert:*")
            for key in alert_keys:
                try:
                    alert_data = self.redis_client.get(key)
                    if alert_data:
                        alert = json.loads(alert_data)
                        alert_time = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
                        if alert_time < cutoff_time:
                            self.redis_client.delete(key)
                except Exception:
                    # If we can't parse the alert, delete it
                    self.redis_client.delete(key)
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def record_trading_signal(self, strategy: str, signal_type: str, asset: str, **metadata):
        """Record a trading signal for monitoring."""
        if self.metrics_collector:
            self.metrics_collector.trading_metrics.record_signal(strategy, signal_type, asset)
            
        if self.structured_logger:
            with correlation_context.context(component=LogComp.TRADING_ENGINE):
                self.structured_logger.log_business_event(
                    "trading_signal_generated",
                    strategy=strategy,
                    signal_type=signal_type,
                    asset=asset,
                    **metadata
                )
                
    def record_order_execution(self, order_type: str, side: str, asset: str, 
                             status: str, execution_time_ms: float, **metadata):
        """Record order execution for monitoring."""
        if self.metrics_collector:
            self.metrics_collector.trading_metrics.record_order(order_type, side, asset, status)
            if status == 'filled':
                self.metrics_collector.trading_metrics.record_execution_latency(
                    order_type, metadata.get('venue', 'default'), execution_time_ms
                )
                
        if self.sla_monitor:
            self.sla_monitor.record_measurement(
                "order_execution", 
                SLAType.RESPONSE_TIME, 
                execution_time_ms
            )
            
        if self.structured_logger:
            with correlation_context.context(component=LogComp.EXECUTION_ENGINE):
                self.structured_logger.log_business_event(
                    "order_executed",
                    order_type=order_type,
                    side=side,
                    asset=asset,
                    status=status,
                    execution_time_ms=execution_time_ms,
                    **metadata
                )
                
    def record_risk_metric(self, portfolio: str, var_95: float, var_99: float, 
                         margin_usage: float, **metadata):
        """Record risk metrics for monitoring."""
        if self.metrics_collector:
            self.metrics_collector.risk_metrics.update_var(portfolio, "1d", var_95, var_99)
            self.metrics_collector.risk_metrics.update_margin_usage("main_account", margin_usage)
            
        # Check for risk alerts
        if var_95 > 0.05:  # 5% VaR threshold
            self.send_alert(
                alert_type="risk_var_breach",
                priority=AlertPriority.CRITICAL,
                title="VaR Threshold Breached",
                message=f"Portfolio VaR ({var_95:.2%}) exceeds 5% threshold",
                source="risk_management",
                metrics={'var_95': var_95, 'threshold': 0.05}
            )
            
        if self.structured_logger:
            with correlation_context.context(component=LogComp.RISK_MANAGEMENT):
                self.structured_logger.log_business_event(
                    "risk_metrics_updated",
                    portfolio=portfolio,
                    var_95=var_95,
                    var_99=var_99,
                    margin_usage=margin_usage,
                    **metadata
                )
                
    def record_marl_inference(self, agent_type: str, model_name: str, 
                            inference_time_ms: float, accuracy: float, **metadata):
        """Record MARL agent inference metrics."""
        if self.metrics_collector:
            self.metrics_collector.marl_metrics.record_inference_time(
                agent_type, model_name, inference_time_ms
            )
            self.metrics_collector.marl_metrics.update_accuracy(
                agent_type, "1h", accuracy
            )
            
        if self.sla_monitor:
            self.sla_monitor.record_measurement(
                "marl_agents", 
                SLAType.RESPONSE_TIME, 
                inference_time_ms
            )
            
        if self.structured_logger:
            with correlation_context.context(component=LogComp.MARL_AGENTS):
                self.structured_logger.log_performance(
                    "marl_inference",
                    inference_time_ms / 1000,  # Convert to seconds
                    agent_type=agent_type,
                    model_name=model_name,
                    accuracy=accuracy,
                    **metadata
                )
                
    def send_alert(self, alert_type: str, priority: AlertPriority, title: str, 
                  message: str, source: str, metrics: Dict[str, Any]):
        """Send alert through the alerting system."""
        if self.alerting_system:
            alert = EnhancedAlert(
                id=f"alert_{int(time.time())}_{alert_type}",
                timestamp=datetime.utcnow(),
                priority=priority,
                status=AlertStatus.ACTIVE,
                source=source,
                alert_type=alert_type,
                title=title,
                message=message,
                metrics=metrics
            )
            
            # Process alert asynchronously
            asyncio.create_task(self.alerting_system.process_alert(alert))
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'running': self.running,
            'component_health': self.component_health,
            'config': {
                'prometheus_enabled': self.config.prometheus_enabled,
                'sla_monitoring_enabled': self.config.sla_monitoring_enabled,
                'alerting_enabled': self.config.alerting_enabled,
                'structured_logging_enabled': self.config.structured_logging_enabled
            }
        }
        
        # Add SLA status if available
        if self.sla_monitor:
            try:
                status['sla_status'] = self.sla_monitor.get_sla_status("trading_engine")
            except Exception as e:
                status['sla_status'] = {'error': str(e)}
                
        # Add alerting status if available
        if self.alerting_system:
            try:
                status['alerting_status'] = asyncio.run(self.alerting_system.get_alert_status())
            except Exception as e:
                status['alerting_status'] = {'error': str(e)}
                
        return status
        
    def run_end_to_end_test(self) -> bool:
        """Run comprehensive end-to-end test of the monitoring system."""
        test_results = {}
        
        try:
            # Test 1: Record trading signal
            self.record_trading_signal("momentum_strategy", "buy", "BTCUSD", confidence=0.85)
            test_results['trading_signal'] = True
            
            # Test 2: Record order execution
            self.record_order_execution(
                "market", "buy", "BTCUSD", "filled", 8.5, 
                venue="binance", quantity=1.0, price=50000.0
            )
            test_results['order_execution'] = True
            
            # Test 3: Record risk metrics
            self.record_risk_metric("main_portfolio", 0.025, 0.035, 65.0, drawdown=0.08)
            test_results['risk_metrics'] = True
            
            # Test 4: Record MARL inference
            self.record_marl_inference(
                "strategic", "transformer", 6.2, 78.5, 
                batch_size=32, model_version="v1.2"
            )
            test_results['marl_inference'] = True
            
            # Test 5: Send test alert
            self.send_alert(
                "test_alert",
                AlertPriority.HIGH,
                "End-to-End Test Alert",
                "This is a test alert for monitoring system validation",
                "monitoring_system",
                {'test': True, 'timestamp': datetime.utcnow().isoformat()}
            )
            test_results['alerting'] = True
            
            # Test 6: Check system status
            status = self.get_system_status()
            test_results['system_status'] = bool(status.get('running'))
            
            # Test 7: Performance test (simulate load)
            start_time = time.time()
            for i in range(100):
                self.record_trading_signal("test_strategy", "buy", "ETHUSDT", test_iteration=i)
            performance_time = time.time() - start_time
            test_results['performance_test'] = performance_time < 5.0  # Should complete in under 5 seconds
            
            # Overall test result
            all_passed = all(test_results.values())
            
            if self.structured_logger:
                self.structured_logger.info(
                    "End-to-end test completed",
                    extra_fields={
                        'test_results': test_results,
                        'all_passed': all_passed,
                        'performance_time': performance_time
                    }
                )
            else:
                logger.info(f"End-to-end test completed: {test_results}")
                
            return all_passed
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            return False

# Factory function
def create_monitoring_system(config: Optional[MonitoringConfig] = None) -> MonitoringSystem:
    """Create monitoring system with optional configuration."""
    if config is None:
        config = MonitoringConfig()
    return MonitoringSystem(config)

# Example usage and testing
if __name__ == "__main__":
    # Create monitoring system
    config = MonitoringConfig(
        prometheus_enabled=True,
        sla_monitoring_enabled=True,
        alerting_enabled=True,
        structured_logging_enabled=True,
        redis_enabled=True
    )
    
    monitoring_system = create_monitoring_system(config)
    
    try:
        # Start monitoring system
        monitoring_system.start()
        
        # Run end-to-end test
        test_passed = monitoring_system.run_end_to_end_test()
        print(f"End-to-end test passed: {test_passed}")
        
        # Get system status
        status = monitoring_system.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")
        
        # Keep running for demonstration
        print("Monitoring system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping monitoring system...")
        monitoring_system.stop()
        print("Monitoring system stopped.")
    except Exception as e:
        print(f"Error: {e}")
        monitoring_system.stop()
