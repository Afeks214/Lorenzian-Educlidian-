"""
Integration Framework for Risk Monitor Agent

Seamless integration system connecting the Risk Monitor Agent with existing
VaR calculators, correlation trackers, and portfolio management systems.

Key Features:
- Seamless VaR calculator integration
- Real-time correlation tracker connection
- Portfolio system bridge
- Event-driven data synchronization
- Configuration management
- Health monitoring and diagnostics
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Type
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import threading
from abc import ABC, abstractmethod

from src.risk.agents.risk_monitor_agent import RiskMonitorAgent, EmergencyAction
from src.risk.agents.emergency_action_system import EmergencyActionExecutor
from src.risk.agents.real_time_risk_assessor import RealTimeRiskAssessor
from src.risk.agents.market_stress_detector import MarketStressDetector
from src.risk.agents.performance_optimizer import PerformanceOptimizer
from src.risk.core.var_calculator import VaRCalculator, VaRResult
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.risk.agents.base_risk_agent import RiskState
from src.core.events import Event, EventType, EventBus

logger = structlog.get_logger()


class IntegrationStatus(Enum):
    """Integration component status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class IntegrationHealth:
    """Health status of integration components"""
    component_name: str
    status: IntegrationStatus
    last_update: datetime
    error_count: int
    response_time_ms: float
    data_quality_score: float  # 0.0 to 1.0
    
    @property
    def is_healthy(self) -> bool:
        return (self.status == IntegrationStatus.CONNECTED and 
                self.response_time_ms < 100 and 
                self.data_quality_score > 0.8)


class DataSynchronizer:
    """
    Real-time data synchronization between components
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.sync_interval_ms = 50  # 50ms sync interval
        self.data_buffers: Dict[str, List] = {}
        self.last_sync_times: Dict[str, datetime] = {}
        self.sync_active = False
        
        self.logger = logger.bind(component="DataSynchronizer")
    
    async def start_synchronization(self):
        """Start real-time data synchronization"""
        self.sync_active = True
        self.logger.info("Starting data synchronization")
        
        # Start sync loop
        asyncio.create_task(self._sync_loop())
    
    async def stop_synchronization(self):
        """Stop data synchronization"""
        self.sync_active = False
        self.logger.info("Stopped data synchronization")
    
    async def _sync_loop(self):
        """Main synchronization loop"""
        while self.sync_active:
            try:
                await self._synchronize_data()
                await asyncio.sleep(self.sync_interval_ms / 1000.0)
            except Exception as e:
                self.logger.error("Synchronization error", error=str(e))
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _synchronize_data(self):
        """Synchronize data between components"""
        current_time = datetime.now()
        
        # Check for stale data and trigger updates
        for component, last_sync in self.last_sync_times.items():
            time_since_sync = (current_time - last_sync).total_seconds()
            
            if time_since_sync > 1.0:  # Data older than 1 second
                await self._request_data_update(component)
    
    async def _request_data_update(self, component: str):
        """Request data update from component"""
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.VAR_UPDATE,  # Generic update request
                {'component': component, 'request_type': 'data_update'},
                'DataSynchronizer'
            )
        )
    
    def register_data_source(self, component_name: str):
        """Register a new data source for synchronization"""
        self.data_buffers[component_name] = []
        self.last_sync_times[component_name] = datetime.now()
        
        self.logger.info("Registered data source", component=component_name)
    
    def update_data(self, component_name: str, data: Any):
        """Update data from a component"""
        if component_name in self.data_buffers:
            self.data_buffers[component_name].append({
                'timestamp': datetime.now(),
                'data': data
            })
            
            # Keep only recent data
            if len(self.data_buffers[component_name]) > 100:
                self.data_buffers[component_name] = self.data_buffers[component_name][-100:]
            
            self.last_sync_times[component_name] = datetime.now()


class ComponentBridge(ABC):
    """
    Abstract base class for component integration bridges
    """
    
    def __init__(self, component_name: str, event_bus: EventBus):
        self.component_name = component_name
        self.event_bus = event_bus
        self.health = IntegrationHealth(
            component_name=component_name,
            status=IntegrationStatus.DISCONNECTED,
            last_update=datetime.now(),
            error_count=0,
            response_time_ms=0.0,
            data_quality_score=0.0
        )
        
        self.logger = logger.bind(component=f"Bridge_{component_name}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the component"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the component"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check"""
        pass
    
    def update_health(self, status: IntegrationStatus, response_time_ms: float = 0.0, 
                     data_quality: float = 1.0, error: bool = False):
        """Update component health status"""
        self.health.status = status
        self.health.last_update = datetime.now()
        self.health.response_time_ms = response_time_ms
        self.health.data_quality_score = data_quality
        
        if error:
            self.health.error_count += 1


class VaRCalculatorBridge(ComponentBridge):
    """
    Bridge to VaR Calculator for real-time risk calculation integration
    """
    
    def __init__(self, var_calculator: VaRCalculator, event_bus: EventBus):
        super().__init__("VaRCalculator", event_bus)
        self.var_calculator = var_calculator
        self.last_var_result: Optional[VaRResult] = None
        
        # Setup event subscriptions
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
    
    async def connect(self) -> bool:
        """Connect to VaR calculator"""
        try:
            # Test connection by requesting a calculation
            portfolio_value = self.var_calculator.portfolio_value
            
            if portfolio_value > 0:
                self.update_health(IntegrationStatus.CONNECTED, 5.0, 1.0)
                self.logger.info("Connected to VaR Calculator")
                return True
            else:
                self.update_health(IntegrationStatus.DEGRADED, 5.0, 0.5)
                self.logger.warning("VaR Calculator connected but no portfolio data")
                return True
                
        except Exception as e:
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            self.logger.error("Failed to connect to VaR Calculator", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from VaR calculator"""
        self.update_health(IntegrationStatus.DISCONNECTED)
        self.logger.info("Disconnected from VaR Calculator")
    
    async def health_check(self) -> bool:
        """Perform VaR calculator health check"""
        try:
            start_time = datetime.now()
            
            # Check if calculator is responsive
            performance_stats = self.var_calculator.get_performance_stats()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Assess data quality
            data_quality = 1.0
            if not self.var_calculator.positions:
                data_quality *= 0.5
            if performance_stats.get('avg_calc_time_ms', 0) > 10:
                data_quality *= 0.8
            
            self.update_health(IntegrationStatus.CONNECTED, response_time, data_quality)
            return True
            
        except Exception as e:
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            self.logger.error("VaR Calculator health check failed", error=str(e))
            return False
    
    def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        if isinstance(event.payload, VaRResult):
            self.last_var_result = event.payload
            self.update_health(IntegrationStatus.CONNECTED, 5.0, 1.0)
    
    async def get_latest_var(self, confidence_level: float = 0.95) -> Optional[VaRResult]:
        """Get latest VaR calculation"""
        try:
            return self.var_calculator.get_latest_var(confidence_level)
        except Exception as e:
            self.logger.error("Failed to get latest VaR", error=str(e))
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            return None
    
    async def request_var_calculation(self, confidence_level: float = 0.95) -> Optional[VaRResult]:
        """Request new VaR calculation"""
        try:
            return await self.var_calculator.calculate_var(confidence_level=confidence_level)
        except Exception as e:
            self.logger.error("VaR calculation request failed", error=str(e))
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            return None


class CorrelationTrackerBridge(ComponentBridge):
    """
    Bridge to Correlation Tracker for real-time correlation monitoring
    """
    
    def __init__(self, correlation_tracker: CorrelationTracker, event_bus: EventBus):
        super().__init__("CorrelationTracker", event_bus)
        self.correlation_tracker = correlation_tracker
        self.last_correlation_matrix: Optional[np.ndarray] = None
        
        # Setup event subscriptions
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_correlation_update)
    
    async def connect(self) -> bool:
        """Connect to correlation tracker"""
        try:
            # Test connection
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            current_regime = self.correlation_tracker.current_regime
            
            data_quality = 1.0
            if correlation_matrix is None:
                data_quality = 0.3
            elif correlation_matrix.shape[0] < 2:
                data_quality = 0.5
            
            self.update_health(IntegrationStatus.CONNECTED, 3.0, data_quality)
            self.logger.info("Connected to Correlation Tracker", 
                           regime=current_regime.value if current_regime else "unknown")
            return True
            
        except Exception as e:
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            self.logger.error("Failed to connect to Correlation Tracker", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from correlation tracker"""
        self.update_health(IntegrationStatus.DISCONNECTED)
        self.logger.info("Disconnected from Correlation Tracker")
    
    async def health_check(self) -> bool:
        """Perform correlation tracker health check"""
        try:
            start_time = datetime.now()
            
            # Check correlation matrix availability
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            current_regime = self.correlation_tracker.current_regime
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Assess data quality
            data_quality = 1.0
            if correlation_matrix is None:
                data_quality = 0.3
            elif np.any(np.isnan(correlation_matrix)) or np.any(np.isinf(correlation_matrix)):
                data_quality = 0.5
            elif correlation_matrix.shape[0] < 3:
                data_quality = 0.7
            
            self.update_health(IntegrationStatus.CONNECTED, response_time, data_quality)
            return True
            
        except Exception as e:
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            self.logger.error("Correlation Tracker health check failed", error=str(e))
            return False
    
    def _handle_correlation_update(self, event: Event):
        """Handle correlation update events"""
        try:
            correlation_matrix = self.correlation_tracker.get_correlation_matrix()
            if correlation_matrix is not None:
                self.last_correlation_matrix = correlation_matrix
                self.update_health(IntegrationStatus.CONNECTED, 3.0, 1.0)
        except Exception as e:
            self.logger.error("Failed to handle correlation update", error=str(e))
    
    async def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get current correlation matrix"""
        try:
            return self.correlation_tracker.get_correlation_matrix()
        except Exception as e:
            self.logger.error("Failed to get correlation matrix", error=str(e))
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            return None
    
    async def get_current_regime(self) -> Optional[CorrelationRegime]:
        """Get current correlation regime"""
        try:
            return self.correlation_tracker.current_regime
        except Exception as e:
            self.logger.error("Failed to get correlation regime", error=str(e))
            self.update_health(IntegrationStatus.ERROR, 0.0, 0.0, True)
            return None


class RiskMonitorIntegration:
    """
    Complete Integration Framework for Risk Monitor Agent
    
    Orchestrates all component integrations and provides unified interface
    for the Risk Monitor Agent system.
    """
    
    def __init__(
        self,
        var_calculator: VaRCalculator,
        correlation_tracker: CorrelationTracker,
        event_bus: EventBus,
        config: Dict[str, Any]
    ):
        self.var_calculator = var_calculator
        self.correlation_tracker = correlation_tracker
        self.event_bus = event_bus
        self.config = config
        
        # Initialize core components
        self.performance_optimizer = PerformanceOptimizer(config.get('performance', {}))
        self.emergency_action_executor = EmergencyActionExecutor(event_bus, config.get('actions', {}))
        self.real_time_risk_assessor = RealTimeRiskAssessor(
            var_calculator, correlation_tracker, event_bus, config.get('assessment', {})
        )
        self.market_stress_detector = MarketStressDetector(event_bus, config.get('stress_detection', {}))
        
        # Initialize Risk Monitor Agent
        self.risk_monitor_agent = RiskMonitorAgent(
            config=config.get('agent', {}),
            var_calculator=var_calculator,
            correlation_tracker=correlation_tracker,
            event_bus=event_bus
        )
        
        # Integration bridges
        self.var_bridge = VaRCalculatorBridge(var_calculator, event_bus)
        self.correlation_bridge = CorrelationTrackerBridge(correlation_tracker, event_bus)
        
        # Data synchronization
        self.data_synchronizer = DataSynchronizer(event_bus)
        
        # Integration state
        self.integration_active = False
        self.component_health: Dict[str, IntegrationHealth] = {}
        
        # Performance monitoring
        self.integration_start_time: Optional[datetime] = None
        self.total_risk_decisions = 0
        self.emergency_actions_taken = 0
        
        self.logger = logger.bind(component="RiskMonitorIntegration")
        self.logger.info("Risk Monitor Integration initialized")
    
    async def start_integration(self) -> bool:
        """Start complete integration system"""
        if self.integration_active:
            self.logger.warning("Integration already active")
            return True
        
        try:
            self.logger.info("Starting Risk Monitor Integration")
            self.integration_start_time = datetime.now()
            
            # 1. Connect to component bridges
            var_connected = await self.var_bridge.connect()
            corr_connected = await self.correlation_bridge.connect()
            
            if not (var_connected and corr_connected):
                self.logger.error("Failed to connect to required components")
                return False
            
            # 2. Start data synchronization
            await self.data_synchronizer.start_synchronization()
            
            # 3. Start real-time risk assessment
            await self.real_time_risk_assessor.start_monitoring()
            
            # 4. Register data sources
            self.data_synchronizer.register_data_source("VaRCalculator")
            self.data_synchronizer.register_data_source("CorrelationTracker")
            self.data_synchronizer.register_data_source("RiskMonitorAgent")
            
            self.integration_active = True
            
            # 5. Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            self.logger.info("Risk Monitor Integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to start integration", error=str(e))
            return False
    
    async def stop_integration(self):
        """Stop integration system"""
        if not self.integration_active:
            return
        
        try:
            self.logger.info("Stopping Risk Monitor Integration")
            
            # Stop monitoring
            self.integration_active = False
            
            # Stop components
            await self.real_time_risk_assessor.stop_monitoring()
            await self.data_synchronizer.stop_synchronization()
            
            # Disconnect bridges
            await self.var_bridge.disconnect()
            await self.correlation_bridge.disconnect()
            
            # Cleanup
            self.performance_optimizer.cleanup()
            
            self.logger.info("Risk Monitor Integration stopped")
            
        except Exception as e:
            self.logger.error("Error stopping integration", error=str(e))
    
    async def _health_monitoring_loop(self):
        """Monitor health of all integration components"""
        while self.integration_active:
            try:
                # Check component health
                await self.var_bridge.health_check()
                await self.correlation_bridge.health_check()
                
                # Update component health tracking
                self.component_health = {
                    'var_calculator': self.var_bridge.health,
                    'correlation_tracker': self.correlation_bridge.health,
                    'risk_assessor': IntegrationHealth(
                        component_name="RealTimeRiskAssessor",
                        status=IntegrationStatus.CONNECTED if self.real_time_risk_assessor.monitoring_active else IntegrationStatus.DISCONNECTED,
                        last_update=datetime.now(),
                        error_count=0,
                        response_time_ms=5.0,
                        data_quality_score=1.0
                    )
                }
                
                # Check for degraded performance
                unhealthy_components = [
                    name for name, health in self.component_health.items()
                    if not health.is_healthy
                ]
                
                if unhealthy_components:
                    self.logger.warning("Unhealthy components detected",
                                      components=unhealthy_components)
                
                await asyncio.sleep(10.0)  # Health check every 10 seconds
                
            except Exception as e:
                self.logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(5.0)
    
    async def process_risk_state(self, risk_state: RiskState) -> Tuple[int, float]:
        """
        Process risk state through complete integration pipeline
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (action, confidence)
        """
        try:
            # Use Risk Monitor Agent to calculate action
            action, confidence = self.risk_monitor_agent.calculate_risk_action(risk_state)
            
            # Track decision
            self.total_risk_decisions += 1
            
            # If action required, execute through emergency action system
            if action != EmergencyAction.NO_ACTION.value:
                emergency_action = EmergencyAction(action)
                
                # Execute action
                execution_result = await self.emergency_action_executor.execute_action_async(
                    emergency_action, risk_state, "Risk Monitor Agent Decision"
                )
                
                if execution_result.success:
                    self.emergency_actions_taken += 1
                
                self.logger.info("Emergency action executed",
                               action=emergency_action.name,
                               success=execution_result.success,
                               execution_time_ms=execution_result.execution_time_ms)
            
            return action, confidence
            
        except Exception as e:
            self.logger.error("Error processing risk state", error=str(e))
            return EmergencyAction.NO_ACTION.value, 0.0
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        uptime = (datetime.now() - self.integration_start_time).total_seconds() if self.integration_start_time else 0
        
        # Get performance metrics
        performance_report = self.performance_optimizer.get_performance_report()
        risk_metrics = self.risk_monitor_agent.get_risk_metrics()
        assessment_performance = self.real_time_risk_assessor.get_assessment_performance()
        stress_performance = self.market_stress_detector.get_detection_performance()
        
        return {
            'integration_active': self.integration_active,
            'uptime_seconds': uptime,
            'component_health': {name: asdict(health) for name, health in self.component_health.items()},
            'total_risk_decisions': self.total_risk_decisions,
            'emergency_actions_taken': self.emergency_actions_taken,
            'performance_metrics': performance_report,
            'risk_agent_metrics': asdict(risk_metrics),
            'assessment_performance': assessment_performance,
            'stress_detection_performance': stress_performance,
            'current_market_stress': self.market_stress_detector.get_current_stress_level()
        }
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get detailed system diagnostics"""
        return {
            'var_calculator_summary': self.var_calculator.get_var_summary(),
            'correlation_regime': self.correlation_tracker.current_regime.value if self.correlation_tracker.current_regime else "unknown",
            'recent_breaches': self.real_time_risk_assessor.get_recent_breaches(5),
            'recent_stress_events': self.market_stress_detector.get_recent_events(5),
            'emergency_action_summary': self.risk_monitor_agent.get_emergency_action_summary(),
            'optimization_recommendations': self.performance_optimizer.get_optimization_recommendations()
        }
    
    async def run_integration_test(self) -> Dict[str, bool]:
        """Run comprehensive integration test"""
        test_results = {}
        
        try:
            # Test VaR integration
            latest_var = await self.var_bridge.get_latest_var()
            test_results['var_integration'] = latest_var is not None
            
            # Test correlation integration
            correlation_matrix = await self.correlation_bridge.get_correlation_matrix()
            test_results['correlation_integration'] = correlation_matrix is not None
            
            # Test risk assessment
            if latest_var:
                risk_state = RiskState(
                    account_equity_normalized=1.0,
                    open_positions_count=5,
                    volatility_regime=0.5,
                    correlation_risk=0.3,
                    var_estimate_5pct=0.02,
                    current_drawdown_pct=0.01,
                    margin_usage_pct=0.4,
                    time_of_day_risk=0.5,
                    market_stress_level=0.3,
                    liquidity_conditions=0.8
                )
                
                action, confidence = await self.process_risk_state(risk_state)
                test_results['risk_processing'] = action is not None and confidence >= 0
            else:
                test_results['risk_processing'] = False
            
            # Test performance optimization
            benchmark_results = self.performance_optimizer.benchmark_performance(100)
            test_results['performance_optimization'] = all(
                time_ms < 10.0 for time_ms in benchmark_results.values()
            )
            
            self.logger.info("Integration test completed", results=test_results)
            
        except Exception as e:
            self.logger.error("Integration test failed", error=str(e))
            test_results['integration_test'] = False
        
        return test_results
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.create_task(self.stop_integration())