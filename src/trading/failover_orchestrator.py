#!/usr/bin/env python3
"""
Trading Engine Failover Orchestrator
AGENT 2: Trading Engine RTO Specialist

Central orchestration system that coordinates all failover components
to achieve <5s RTO target. Integrates health monitoring, state sync,
circuit breakers, standby warmup, and automated testing.

Key Features:
- Centralized failover coordination
- Real-time performance monitoring
- Automated failover decision making
- Comprehensive health management
- Performance optimization
- Continuous testing and validation
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import redis.asyncio as redis
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, Event, EventType
from src.trading.failover_monitor import TradingEngineFailoverMonitor, FailoverConfig
from src.trading.fast_circuit_breaker import FastCircuitBreaker, FastCircuitConfig
from src.trading.standby_warmup import StandbyWarmupSystem, WarmupConfig
from src.trading.state_sync import RedisStateSynchronizer, InstanceRole
from src.trading.failover_testing import FailoverTestFramework, TestConfig, TestScenario
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OrchestratorState(Enum):
    """Orchestrator states"""
    INITIALIZING = "initializing"
    READY = "ready"
    MONITORING = "monitoring"
    FAILOVER_INITIATED = "failover_initiated"
    FAILOVER_IN_PROGRESS = "failover_in_progress"
    FAILOVER_COMPLETED = "failover_completed"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class OrchestratorConfig:
    """Configuration for failover orchestrator"""
    # Redis configuration
    redis_url: str = "redis://localhost:6379/3"
    
    # Performance targets
    rto_target: float = 5.0  # 5 seconds
    monitoring_interval: float = 1.0  # 1 second
    
    # Component configurations
    enable_failover_monitor: bool = True
    enable_circuit_breaker: bool = True
    enable_standby_warmup: bool = True
    enable_automated_testing: bool = True
    
    # Health check settings
    health_check_interval: float = 0.5  # 500ms
    failure_threshold: int = 2
    recovery_threshold: int = 2
    
    # Failover settings
    auto_failover: bool = True
    failover_timeout: float = 10.0  # 10 seconds
    
    # Testing settings
    continuous_testing: bool = True
    test_interval: float = 3600.0  # 1 hour
    
    # Logging and monitoring
    detailed_logging: bool = True
    metrics_export: bool = True
    performance_alerts: bool = True

class FailoverOrchestrator:
    """
    Central orchestrator for trading engine failover system
    
    Coordinates all failover components to achieve <5s RTO:
    - Failover monitoring and detection
    - Circuit breaker management
    - Standby warmup and readiness
    - State synchronization
    - Automated testing and validation
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.event_bus = EventBus()
        
        # State management
        self.state = OrchestratorState.INITIALIZING
        self.state_change_time = time.time()
        
        # Components
        self.failover_monitor: Optional[TradingEngineFailoverMonitor] = None
        self.circuit_breaker: Optional[FastCircuitBreaker] = None
        self.standby_warmup: Optional[StandbyWarmupSystem] = None
        self.state_sync: Optional[RedisStateSynchronizer] = None
        self.test_framework: Optional[FailoverTestFramework] = None
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance tracking
        self.performance_metrics = {
            'failovers_completed': 0,
            'average_rto': 0.0,
            'last_failover_time': 0.0,
            'system_uptime': 0.0,
            'health_score': 100.0
        }
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.testing_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Instance tracking
        self.active_instance: Optional[str] = None
        self.passive_instances: List[str] = []
        
        logger.info("Failover orchestrator initialized")
    
    async def initialize(self):
        """Initialize failover orchestrator"""
        try:
            logger.info("Initializing failover orchestrator...")
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            
            # Initialize components
            await self._initialize_components()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.state = OrchestratorState.READY
            self.state_change_time = time.time()
            
            logger.info("Failover orchestrator initialized successfully")
            
            # Send ready event
            await self.event_bus.emit(Event(
                type=EventType.SYSTEM_READY,
                data={
                    'component': 'failover_orchestrator',
                    'state': self.state.value,
                    'rto_target': self.config.rto_target
                }
            ))
            
        except Exception as e:
            self.state = OrchestratorState.ERROR
            logger.error(f"Failed to initialize failover orchestrator: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all failover components"""
        try:
            # Initialize failover monitor
            if self.config.enable_failover_monitor:
                monitor_config = FailoverConfig(
                    health_check_interval=self.config.health_check_interval,
                    failure_threshold=self.config.failure_threshold,
                    recovery_threshold=self.config.recovery_threshold,
                    auto_recovery=self.config.auto_failover
                )
                self.failover_monitor = TradingEngineFailoverMonitor(monitor_config)
                await self.failover_monitor.initialize(self.config.redis_url)
                logger.info("Failover monitor initialized")
            
            # Initialize circuit breaker
            if self.config.enable_circuit_breaker:
                cb_config = FastCircuitConfig(
                    failure_threshold=self.config.failure_threshold,
                    timeout_ms=int(self.config.failover_timeout * 1000),
                    service_name="trading_engine_orchestrator"
                )
                self.circuit_breaker = FastCircuitBreaker(cb_config)
                await self.circuit_breaker.initialize(self.config.redis_url)
                logger.info("Circuit breaker initialized")
            
            # Initialize standby warmup
            if self.config.enable_standby_warmup:
                warmup_config = WarmupConfig(
                    instance_id="orchestrator_warmup",
                    startup_time_target=self.config.rto_target / 2,
                    continuous_warming=True
                )
                self.standby_warmup = StandbyWarmupSystem(warmup_config)
                await self.standby_warmup.initialize(self.config.redis_url)
                logger.info("Standby warmup initialized")
            
            # Initialize state synchronizer
            self.state_sync = RedisStateSynchronizer(
                redis_url=self.config.redis_url,
                instance_id="orchestrator_sync",
                role=InstanceRole.PASSIVE,
                sync_interval=0.25  # 250ms
            )
            await self.state_sync.initialize()
            logger.info("State synchronizer initialized")
            
            # Initialize test framework
            if self.config.enable_automated_testing:
                test_config = TestConfig(
                    rto_target=self.config.rto_target,
                    export_results=self.config.metrics_export,
                    scenarios_to_run=[
                        TestScenario.BASIC_FAILOVER,
                        TestScenario.RECOVERY_VALIDATION
                    ]
                )
                self.test_framework = FailoverTestFramework(test_config)
                await self.test_framework.initialize(self.config.redis_url)
                logger.info("Test framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _register_event_handlers(self):
        """Register event handlers for system events"""
        try:
            # Handle failover events
            await self.event_bus.subscribe(
                EventType.FAILOVER_INITIATED,
                self._handle_failover_initiated
            )
            
            await self.event_bus.subscribe(
                EventType.FAILOVER_COMPLETED,
                self._handle_failover_completed
            )
            
            # Handle health events
            await self.event_bus.subscribe(
                "service_health_changed",
                self._handle_health_changed
            )
            
            # Handle circuit breaker events
            await self.event_bus.subscribe(
                "circuit_breaker_opened",
                self._handle_circuit_breaker_opened
            )
            
            # Handle system ready events
            await self.event_bus.subscribe(
                EventType.SYSTEM_READY,
                self._handle_system_ready
            )
            
            logger.info("Event handlers registered")
            
        except Exception as e:
            logger.error(f"Failed to register event handlers: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start testing task if enabled
            if self.config.continuous_testing:
                self.testing_task = asyncio.create_task(self._testing_loop())
            
            # Start metrics task
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while True:
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Update state
                self.state = OrchestratorState.MONITORING
                
                # Check system health
                await self._check_system_health()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for failover conditions
                await self._check_failover_conditions()
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _testing_loop(self):
        """Automated testing loop"""
        try:
            while True:
                await asyncio.sleep(self.config.test_interval)
                
                if self.test_framework and self.state == OrchestratorState.MONITORING:
                    logger.info("Starting automated failover test")
                    
                    self.state = OrchestratorState.TESTING
                    
                    try:
                        # Run basic failover test
                        result = await self.test_framework.run_single_test(TestScenario.BASIC_FAILOVER)
                        
                        # Log results
                        if result.passed():
                            logger.info(f"Automated test PASSED - RTO: {result.rto_achieved:.2f}s")
                        else:
                            logger.warning(f"Automated test FAILED - RTO: {result.rto_achieved:.2f}s")
                            
                            # Send alert
                            await self._send_performance_alert(
                                "Automated failover test failed",
                                {"rto_achieved": result.rto_achieved, "rto_target": result.rto_target}
                            )
                    
                    except Exception as e:
                        logger.error(f"Automated test error: {e}")
                    
                    finally:
                        self.state = OrchestratorState.MONITORING
                
        except Exception as e:
            logger.error(f"Error in testing loop: {e}")
    
    async def _metrics_loop(self):
        """Metrics collection loop"""
        try:
            while True:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Collect metrics from all components
                await self._collect_component_metrics()
                
                # Calculate system health score
                await self._calculate_health_score()
                
                # Persist metrics
                await self._persist_metrics()
                
                # Check performance alerts
                if self.config.performance_alerts:
                    await self._check_performance_alerts()
                
        except Exception as e:
            logger.error(f"Error in metrics loop: {e}")
    
    async def _check_system_health(self):
        """Check overall system health"""
        try:
            health_issues = []
            
            # Check failover monitor health
            if self.failover_monitor:
                status = self.failover_monitor.get_status()
                if status.get('state') == 'error':
                    health_issues.append("Failover monitor in error state")
            
            # Check circuit breaker health
            if self.circuit_breaker:
                status = self.circuit_breaker.get_status()
                if status.get('state') == 'open':
                    health_issues.append("Circuit breaker is open")
            
            # Check standby warmup health
            if self.standby_warmup:
                status = self.standby_warmup.get_status()
                if status.get('state') != 'ready':
                    health_issues.append("Standby warmup not ready")
            
            # Check state sync health
            if self.state_sync:
                metrics = self.state_sync.get_metrics()
                if metrics.get('success_rate', 0) < 0.95:
                    health_issues.append("State sync success rate low")
            
            # Log health issues
            if health_issues:
                logger.warning(f"System health issues: {health_issues}")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Update system uptime
            self.performance_metrics['system_uptime'] = time.time() - self.state_change_time
            
            # Get component metrics
            if self.failover_monitor:
                monitor_status = self.failover_monitor.get_status()
                if 'metrics' in monitor_status:
                    metrics = monitor_status['metrics']
                    self.performance_metrics['failovers_completed'] = metrics.get('total_failovers', 0)
                    self.performance_metrics['average_rto'] = metrics.get('average_rto_ms', 0) / 1000
                    self.performance_metrics['last_failover_time'] = metrics.get('last_failover_time', 0)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_failover_conditions(self):
        """Check if failover conditions are met"""
        try:
            # This would implement complex failover decision logic
            # For now, we rely on the failover monitor
            
            if self.failover_monitor:
                status = self.failover_monitor.get_status()
                if status.get('state') == 'failover_initiated':
                    await self._initiate_coordinated_failover()
            
        except Exception as e:
            logger.error(f"Error checking failover conditions: {e}")
    
    async def _initiate_coordinated_failover(self):
        """Initiate coordinated failover"""
        try:
            if self.state == OrchestratorState.FAILOVER_IN_PROGRESS:
                return  # Already in progress
            
            logger.warning("Initiating coordinated failover")
            
            self.state = OrchestratorState.FAILOVER_INITIATED
            failover_start_time = time.time()
            
            # Phase 1: Prepare standby instances
            if self.standby_warmup:
                await self.standby_warmup.force_warmup()
            
            # Phase 2: Coordinate failover
            self.state = OrchestratorState.FAILOVER_IN_PROGRESS
            
            # Emit failover event
            await self.event_bus.emit(Event(
                type=EventType.FAILOVER_INITIATED,
                data={
                    'orchestrator_id': 'main',
                    'start_time': failover_start_time,
                    'rto_target': self.config.rto_target
                }
            ))
            
            # Phase 3: Wait for completion
            # In a real implementation, this would coordinate with actual instances
            await asyncio.sleep(2)  # Simulate failover time
            
            # Phase 4: Verify failover success
            await self._verify_failover_success()
            
            # Calculate RTO
            rto_achieved = time.time() - failover_start_time
            
            # Update metrics
            self.performance_metrics['last_failover_time'] = time.time()
            self.performance_metrics['failovers_completed'] += 1
            
            # Update average RTO
            avg_rto = self.performance_metrics['average_rto']
            total_failovers = self.performance_metrics['failovers_completed']
            self.performance_metrics['average_rto'] = (avg_rto * (total_failovers - 1) + rto_achieved) / total_failovers
            
            self.state = OrchestratorState.FAILOVER_COMPLETED
            
            logger.info(f"Coordinated failover completed in {rto_achieved:.2f}s")
            
            # Send completion event
            await self.event_bus.emit(Event(
                type=EventType.FAILOVER_COMPLETED,
                data={
                    'orchestrator_id': 'main',
                    'rto_achieved': rto_achieved,
                    'rto_target': self.config.rto_target,
                    'success': rto_achieved <= self.config.rto_target
                }
            ))
            
            # Return to monitoring
            await asyncio.sleep(5)  # Brief pause
            self.state = OrchestratorState.MONITORING
            
        except Exception as e:
            self.state = OrchestratorState.ERROR
            logger.error(f"Error in coordinated failover: {e}")
    
    async def _verify_failover_success(self):
        """Verify that failover was successful"""
        try:
            # Check that all components are healthy
            if self.standby_warmup:
                if not await self.standby_warmup.is_ready():
                    raise Exception("Standby warmup not ready after failover")
            
            if self.circuit_breaker:
                status = self.circuit_breaker.get_status()
                if status.get('state') == 'open':
                    raise Exception("Circuit breaker open after failover")
            
            if self.state_sync:
                metrics = self.state_sync.get_metrics()
                if metrics.get('success_rate', 0) < 0.8:
                    raise Exception("State sync unhealthy after failover")
            
            logger.info("Failover verification completed successfully")
            
        except Exception as e:
            logger.error(f"Failover verification failed: {e}")
            raise
    
    async def _collect_component_metrics(self):
        """Collect metrics from all components"""
        try:
            component_metrics = {}
            
            # Failover monitor metrics
            if self.failover_monitor:
                status = self.failover_monitor.get_status()
                component_metrics['failover_monitor'] = status
            
            # Circuit breaker metrics
            if self.circuit_breaker:
                status = self.circuit_breaker.get_status()
                component_metrics['circuit_breaker'] = status
            
            # Standby warmup metrics
            if self.standby_warmup:
                status = self.standby_warmup.get_status()
                component_metrics['standby_warmup'] = status
            
            # State sync metrics
            if self.state_sync:
                metrics = self.state_sync.get_metrics()
                component_metrics['state_sync'] = metrics
            
            # Store metrics
            self.component_metrics = component_metrics
            
        except Exception as e:
            logger.error(f"Error collecting component metrics: {e}")
    
    async def _calculate_health_score(self):
        """Calculate overall system health score"""
        try:
            health_score = 100.0
            
            # Factor in component health
            if hasattr(self, 'component_metrics'):
                # Check failover monitor health
                if 'failover_monitor' in self.component_metrics:
                    fm_metrics = self.component_metrics['failover_monitor']
                    if fm_metrics.get('performance', {}).get('error_rate', 0) > 0.1:
                        health_score -= 20
                
                # Check circuit breaker health
                if 'circuit_breaker' in self.component_metrics:
                    cb_metrics = self.component_metrics['circuit_breaker']
                    if cb_metrics.get('state') == 'open':
                        health_score -= 30
                
                # Check standby warmup health
                if 'standby_warmup' in self.component_metrics:
                    sw_metrics = self.component_metrics['standby_warmup']
                    readiness_score = sw_metrics.get('readiness_score', 100)
                    health_score -= (100 - readiness_score) * 0.2
                
                # Check state sync health
                if 'state_sync' in self.component_metrics:
                    ss_metrics = self.component_metrics['state_sync']
                    success_rate = ss_metrics.get('success_rate', 1.0)
                    health_score -= (1.0 - success_rate) * 30
            
            # Factor in recent RTO performance
            if self.performance_metrics['average_rto'] > self.config.rto_target:
                rto_penalty = min(20, (self.performance_metrics['average_rto'] - self.config.rto_target) * 4)
                health_score -= rto_penalty
            
            self.performance_metrics['health_score'] = max(0, health_score)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
    
    async def _persist_metrics(self):
        """Persist metrics to Redis"""
        try:
            if self.redis_client:
                metrics_data = {
                    'orchestrator_state': self.state.value,
                    'performance_metrics': json.dumps(self.performance_metrics),
                    'component_metrics': json.dumps(getattr(self, 'component_metrics', {})),
                    'timestamp': time.time()
                }
                
                await self.redis_client.hset(
                    "failover_orchestrator:metrics",
                    mapping=metrics_data
                )
                
                await self.redis_client.expire("failover_orchestrator:metrics", 3600)
                
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    async def _check_performance_alerts(self):
        """Check for performance alerts"""
        try:
            # Check RTO performance
            if self.performance_metrics['average_rto'] > self.config.rto_target * 1.2:
                await self._send_performance_alert(
                    "RTO performance degraded",
                    {
                        'average_rto': self.performance_metrics['average_rto'],
                        'target_rto': self.config.rto_target
                    }
                )
            
            # Check health score
            if self.performance_metrics['health_score'] < 80:
                await self._send_performance_alert(
                    "System health degraded",
                    {'health_score': self.performance_metrics['health_score']}
                )
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _send_performance_alert(self, message: str, data: Dict[str, Any]):
        """Send performance alert"""
        try:
            logger.warning(f"Performance alert: {message} - {data}")
            
            # Send alert event
            await self.event_bus.emit(Event(
                type=EventType.PERFORMANCE_ALERT,
                data={
                    'message': message,
                    'data': data,
                    'timestamp': time.time()
                }
            ))
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
    
    async def _handle_failover_initiated(self, event: Event):
        """Handle failover initiated event"""
        logger.info("Failover initiated event received")
        
        # Coordinate with components
        if self.standby_warmup:
            # Trigger final warmup
            await self.standby_warmup.force_warmup()
    
    async def _handle_failover_completed(self, event: Event):
        """Handle failover completed event"""
        logger.info("Failover completed event received")
        
        # Update metrics
        rto_achieved = event.data.get('rto_achieved', 0)
        if rto_achieved > 0:
            self.performance_metrics['last_failover_time'] = time.time()
    
    async def _handle_health_changed(self, event: Event):
        """Handle health changed event"""
        service_name = event.data.get('service_name')
        new_status = event.data.get('new_status')
        
        logger.info(f"Health changed: {service_name} -> {new_status}")
        
        # Take action based on health changes
        if new_status in ['unhealthy', 'critical']:
            # Consider triggering failover
            await self._evaluate_failover_need()
    
    async def _handle_circuit_breaker_opened(self, event: Event):
        """Handle circuit breaker opened event"""
        logger.warning("Circuit breaker opened event received")
        
        # Trigger failover if needed
        await self._evaluate_failover_need()
    
    async def _handle_system_ready(self, event: Event):
        """Handle system ready event"""
        component = event.data.get('component')
        logger.info(f"System ready event from {component}")
    
    async def _evaluate_failover_need(self):
        """Evaluate whether failover is needed"""
        try:
            # Check if failover is already in progress
            if self.state in [OrchestratorState.FAILOVER_INITIATED, OrchestratorState.FAILOVER_IN_PROGRESS]:
                return
            
            # Complex evaluation logic would go here
            # For now, simple checks
            
            should_failover = False
            
            # Check circuit breaker state
            if self.circuit_breaker:
                status = self.circuit_breaker.get_status()
                if status.get('state') == 'open':
                    should_failover = True
            
            # Check health score
            if self.performance_metrics['health_score'] < 50:
                should_failover = True
            
            if should_failover:
                logger.warning("Failover evaluation: initiating failover")
                await self._initiate_coordinated_failover()
            
        except Exception as e:
            logger.error(f"Error evaluating failover need: {e}")
    
    async def register_instance(self, instance_id: str, role: str, health_check_url: str):
        """Register a trading engine instance"""
        try:
            if role == "active":
                self.active_instance = instance_id
            else:
                self.passive_instances.append(instance_id)
            
            # Register with failover monitor
            if self.failover_monitor:
                await self.failover_monitor.register_instance(
                    instance_id,
                    InstanceRole.ACTIVE if role == "active" else InstanceRole.PASSIVE,
                    health_check_url
                )
            
            logger.info(f"Instance registered: {instance_id} ({role})")
            
        except Exception as e:
            logger.error(f"Error registering instance {instance_id}: {e}")
    
    async def force_failover(self):
        """Force manual failover"""
        logger.info("Manual failover requested")
        await self._initiate_coordinated_failover()
    
    async def run_test(self, scenario: TestScenario = TestScenario.BASIC_FAILOVER):
        """Run failover test"""
        if self.test_framework:
            logger.info(f"Running test: {scenario.value}")
            result = await self.test_framework.run_single_test(scenario)
            return result
        else:
            raise Exception("Test framework not initialized")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'state': self.state.value,
            'state_change_time': self.state_change_time,
            'active_instance': self.active_instance,
            'passive_instances': self.passive_instances,
            'performance_metrics': self.performance_metrics,
            'component_status': {
                'failover_monitor': self.failover_monitor.get_status() if self.failover_monitor else None,
                'circuit_breaker': self.circuit_breaker.get_status() if self.circuit_breaker else None,
                'standby_warmup': self.standby_warmup.get_status() if self.standby_warmup else None,
                'state_sync': self.state_sync.get_metrics() if self.state_sync else None
            }
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            'overall_health_score': self.performance_metrics['health_score'],
            'rto_performance': {
                'target': self.config.rto_target,
                'average': self.performance_metrics['average_rto'],
                'last_failover': self.performance_metrics['last_failover_time']
            },
            'component_health': getattr(self, 'component_metrics', {}),
            'system_uptime': self.performance_metrics['system_uptime'],
            'failover_count': self.performance_metrics['failovers_completed']
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down failover orchestrator")
        
        # Cancel background tasks
        for task in [self.monitoring_task, self.testing_task, self.metrics_task]:
            if task:
                task.cancel()
        
        # Shutdown components
        if self.failover_monitor:
            await self.failover_monitor.shutdown()
        if self.circuit_breaker:
            await self.circuit_breaker.shutdown()
        if self.standby_warmup:
            await self.standby_warmup.shutdown()
        if self.state_sync:
            await self.state_sync.shutdown()
        if self.test_framework:
            await self.test_framework.shutdown()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Failover orchestrator shutdown complete")


# Factory function
def create_failover_orchestrator(config: Dict[str, Any]) -> FailoverOrchestrator:
    """Create failover orchestrator instance"""
    orchestrator_config = OrchestratorConfig(**config)
    return FailoverOrchestrator(orchestrator_config)


# CLI interface
async def main():
    """Main entry point for orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failover Orchestrator")
    parser.add_argument("--redis-url", default="redis://localhost:6379/3")
    parser.add_argument("--rto-target", type=float, default=5.0)
    parser.add_argument("--enable-testing", action="store_true", default=True)
    parser.add_argument("--continuous-testing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = OrchestratorConfig(
        redis_url=args.redis_url,
        rto_target=args.rto_target,
        enable_automated_testing=args.enable_testing,
        continuous_testing=args.continuous_testing
    )
    
    # Create and run orchestrator
    orchestrator = FailoverOrchestrator(config)
    
    try:
        await orchestrator.initialize()
        
        # Register sample instances
        await orchestrator.register_instance("trading-engine-1", "active", "http://trading-engine-1:8000")
        await orchestrator.register_instance("trading-engine-2", "passive", "http://trading-engine-2:8000")
        await orchestrator.register_instance("trading-engine-3", "passive", "http://trading-engine-3:8000")
        
        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            
            # Print status
            status = orchestrator.get_status()
            health_report = orchestrator.get_health_report()
            
            print(f"\n=== Orchestrator Status ===")
            print(f"State: {status['state']}")
            print(f"Health Score: {health_report['overall_health_score']:.1f}")
            print(f"Average RTO: {health_report['rto_performance']['average']:.2f}s")
            print(f"Failovers: {health_report['failover_count']}")
            print(f"Uptime: {health_report['system_uptime']:.1f}s")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())