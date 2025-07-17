"""
Strategic MARL Integration for XAI Pipeline

Agent Beta: Real-time streaming specialist
Mission: Seamless integration with existing Strategic MARL system

This module provides seamless integration between the XAI real-time explanation
pipeline and the existing Strategic MARL system, ensuring zero-latency impact
on trading decisions while capturing comprehensive context for explanations.

Key Features:
- Zero-latency integration with Strategic MARL events
- Automatic pipeline initialization and coordination
- Graceful degradation when XAI components are unavailable
- Performance monitoring and health checks
- Configuration management and component lifecycle

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Strategic MARL Integration
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading

from ...core.events import EventType, Event, EventBus
from ...core.component_base import ComponentBase
from .decision_capture import DecisionCapture
from .context_processor import ContextProcessor
from .websocket_manager import WebSocketManager
from .streaming_engine import StreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class IntegrationStatus:
    """Status of XAI pipeline integration"""
    
    pipeline_active: bool = False
    components_initialized: int = 0
    total_components: int = 4
    
    decision_capture_ready: bool = False
    context_processor_ready: bool = False
    websocket_manager_ready: bool = False
    streaming_engine_ready: bool = False
    
    integration_latency_ns: int = 0
    total_decisions_processed: int = 0
    pipeline_errors: int = 0
    
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


class XAIPipelineIntegration(ComponentBase):
    """
    XAI Pipeline Integration with Strategic MARL
    
    Provides seamless integration between the real-time explanation pipeline
    and the existing Strategic MARL system with zero performance impact.
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XAI Pipeline Integration
        
        Args:
            kernel: Reference to the AlgoSpace kernel
            config: Configuration dictionary
        """
        super().__init__("XAIPipelineIntegration", kernel)
        
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('xai.integration')
        
        # Pipeline components
        self.decision_capture: Optional[DecisionCapture] = None
        self.context_processor: Optional[ContextProcessor] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self.streaming_engine: Optional[StreamingEngine] = None
        
        # Integration status
        self.status = IntegrationStatus()
        self.status_lock = threading.Lock()
        
        # Component initialization order and dependencies
        self.component_order = [
            'websocket_manager',
            'decision_capture',
            'context_processor',
            'streaming_engine'
        ]
        
        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = self.config['health_check_interval_seconds']
        
        # Graceful degradation settings
        self.enable_graceful_degradation = self.config['enable_graceful_degradation']
        self.component_failures: Dict[str, int] = {}
        self.max_component_failures = self.config['max_component_failures']
        
        self.logger.info(
            f"XAIPipelineIntegration initialized: "
            f"graceful_degradation={self.enable_graceful_degradation}, "
            f"health_check_interval={self.health_check_interval}s"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Integration Settings
            'enable_graceful_degradation': True,
            'max_component_failures': 3,
            'auto_restart_components': True,
            'restart_delay_seconds': 30,
            
            # Health Monitoring
            'health_check_interval_seconds': 30,
            'component_timeout_seconds': 10,
            'pipeline_latency_threshold_ms': 100,
            
            # Component Configuration
            'decision_capture': {
                'max_capture_latency_ns': 100_000,  # 100 microseconds
                'queue_size': 10000,
                'redis': {'enabled': True}
            },
            
            'context_processor': {
                'queue_size': 5000,
                'embedding_dim': 384,
                'cache_size': 2000
            },
            
            'websocket_manager': {
                'host': '0.0.0.0',
                'port': 8765,
                'max_connections': 1000,
                'authentication': {'enabled': True}
            },
            
            'streaming_engine': {
                'target_explanation_latency_ms': 200,
                'llm': {
                    'model': 'llama3.2:3b',
                    'timeout_seconds': 10
                }
            },
            
            # Performance Monitoring
            'monitoring': {
                'metrics_collection_enabled': True,
                'performance_alerts_enabled': True,
                'log_level': 'INFO'
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the XAI Pipeline Integration"""
        try:
            self.logger.info("Initializing XAI Pipeline Integration...")
            
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Start health monitoring
            if self.config['monitoring']['metrics_collection_enabled']:
                self.health_check_task = asyncio.create_task(self._health_monitor())
            
            # Update status
            with self.status_lock:
                self.status.pipeline_active = True
                self.status.last_health_check = datetime.now(timezone.utc)
                self.status.health_status = "healthy"
            
            self._initialized = True
            self.logger.info("XAI Pipeline Integration initialized successfully")
            
            # Publish integration ready event
            event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {
                    "component": self.name,
                    "status": "ready",
                    "pipeline_active": True,
                    "components_ready": self.status.components_initialized
                },
                source=self.name
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize XAI Pipeline Integration: {e}")
            
            # Enable graceful degradation
            if self.enable_graceful_degradation:
                self.logger.warning("Continuing with graceful degradation mode")
                self._initialized = True
            else:
                raise
    
    async def _initialize_components(self) -> None:
        """Initialize pipeline components in dependency order"""
        
        for component_name in self.component_order:
            try:
                component_config = self.config.get(component_name, {})
                
                if component_name == 'websocket_manager':
                    self.websocket_manager = WebSocketManager(self.kernel, component_config)
                    await self.websocket_manager.initialize()
                    with self.status_lock:
                        self.status.websocket_manager_ready = True
                        self.status.components_initialized += 1
                    
                elif component_name == 'decision_capture':
                    self.decision_capture = DecisionCapture(self.kernel, component_config)
                    await self.decision_capture.initialize()
                    with self.status_lock:
                        self.status.decision_capture_ready = True
                        self.status.components_initialized += 1
                    
                elif component_name == 'context_processor':
                    self.context_processor = ContextProcessor(self.kernel, component_config)
                    await self.context_processor.initialize()
                    with self.status_lock:
                        self.status.context_processor_ready = True
                        self.status.components_initialized += 1
                    
                elif component_name == 'streaming_engine':
                    # Streaming engine needs WebSocket manager
                    if self.websocket_manager:
                        # Update config with WebSocket manager reference
                        component_config['websocket_manager'] = self.websocket_manager
                    
                    self.streaming_engine = StreamingEngine(self.kernel, component_config)
                    await self.streaming_engine.initialize()
                    with self.status_lock:
                        self.status.streaming_engine_ready = True
                        self.status.components_initialized += 1
                
                self.logger.info(f"Component {component_name} initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {component_name}: {e}")
                
                # Track component failures
                self.component_failures[component_name] = self.component_failures.get(component_name, 0) + 1
                
                if not self.enable_graceful_degradation:
                    raise
                
                # Continue with graceful degradation
                self.logger.warning(f"Continuing without {component_name} (graceful degradation)")
    
    async def _health_monitor(self) -> None:
        """Monitor component health and performance"""
        while self._initialized:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Perform health checks
                health_results = await self._perform_health_checks()
                
                # Update status
                with self.status_lock:
                    self.status.last_health_check = datetime.now(timezone.utc)
                    
                    # Determine overall health
                    healthy_components = sum(1 for result in health_results.values() if result.get('healthy', False))
                    total_components = len(health_results)
                    
                    if healthy_components == total_components:
                        self.status.health_status = "healthy"
                    elif healthy_components >= total_components // 2:
                        self.status.health_status = "degraded"
                    else:
                        self.status.health_status = "unhealthy"
                
                # Log health status
                self.logger.debug(
                    f"Health check complete: {healthy_components}/{total_components} components healthy"
                )
                
                # Handle component failures
                await self._handle_component_failures(health_results)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all components"""
        health_results = {}
        
        # Check decision capture
        if self.decision_capture:
            try:
                metrics = self.decision_capture.get_metrics()
                health_results['decision_capture'] = {
                    'healthy': metrics['system_status']['active'],
                    'avg_latency_ns': metrics.get('avg_capture_latency_ns', 0),
                    'error_rate': metrics.get('processing_errors', 0) / max(1, metrics.get('total_decisions_captured', 1))
                }
            except Exception as e:
                health_results['decision_capture'] = {'healthy': False, 'error': str(e)}
        
        # Check context processor
        if self.context_processor:
            try:
                metrics = self.context_processor.get_metrics()
                health_results['context_processor'] = {
                    'healthy': metrics['system_status']['active'],
                    'avg_processing_time_ms': metrics.get('avg_processing_time_ms', 0),
                    'cache_hit_rate': metrics.get('cache_hit_rate', 0)
                }
            except Exception as e:
                health_results['context_processor'] = {'healthy': False, 'error': str(e)}
        
        # Check WebSocket manager
        if self.websocket_manager:
            try:
                metrics = self.websocket_manager.get_metrics()
                health_results['websocket_manager'] = {
                    'healthy': metrics['system_status']['server_running'],
                    'active_connections': metrics.get('active_connections', 0),
                    'message_failure_rate': metrics.get('total_messages_failed', 0) / max(1, metrics.get('total_messages_sent', 1))
                }
            except Exception as e:
                health_results['websocket_manager'] = {'healthy': False, 'error': str(e)}
        
        # Check streaming engine
        if self.streaming_engine:
            try:
                metrics = self.streaming_engine.get_metrics()
                health_results['streaming_engine'] = {
                    'healthy': metrics['system_status']['active'],
                    'avg_generation_time_ms': metrics.get('avg_generation_time_ms', 0),
                    'llm_failure_rate': metrics.get('llm_failures', 0) / max(1, metrics.get('llm_requests', 1))
                }
            except Exception as e:
                health_results['streaming_engine'] = {'healthy': False, 'error': str(e)}
        
        return health_results
    
    async def _handle_component_failures(self, health_results: Dict[str, Dict[str, Any]]) -> None:
        """Handle component failures and restart if configured"""
        
        for component_name, health_data in health_results.items():
            if not health_data.get('healthy', False):
                failure_count = self.component_failures.get(component_name, 0) + 1
                self.component_failures[component_name] = failure_count
                
                self.logger.warning(
                    f"Component {component_name} unhealthy (failure count: {failure_count}): "
                    f"{health_data.get('error', 'unknown error')}"
                )
                
                # Auto-restart if configured and not too many failures
                if (self.config['auto_restart_components'] and 
                    failure_count <= self.max_component_failures):
                    
                    self.logger.info(f"Attempting to restart {component_name}...")
                    await self._restart_component(component_name)
            else:
                # Reset failure count on successful health check
                if component_name in self.component_failures:
                    del self.component_failures[component_name]
    
    async def _restart_component(self, component_name: str) -> None:
        """Restart a failed component"""
        try:
            # Wait before restart
            await asyncio.sleep(self.config['restart_delay_seconds'])
            
            component_config = self.config.get(component_name, {})
            
            if component_name == 'decision_capture' and self.decision_capture:
                await self.decision_capture.shutdown()
                self.decision_capture = DecisionCapture(self.kernel, component_config)
                await self.decision_capture.initialize()
                
            elif component_name == 'context_processor' and self.context_processor:
                await self.context_processor.shutdown()
                self.context_processor = ContextProcessor(self.kernel, component_config)
                await self.context_processor.initialize()
                
            elif component_name == 'websocket_manager' and self.websocket_manager:
                await self.websocket_manager.shutdown()
                self.websocket_manager = WebSocketManager(self.kernel, component_config)
                await self.websocket_manager.initialize()
                
            elif component_name == 'streaming_engine' and self.streaming_engine:
                await self.streaming_engine.shutdown()
                if self.websocket_manager:
                    component_config['websocket_manager'] = self.websocket_manager
                self.streaming_engine = StreamingEngine(self.kernel, component_config)
                await self.streaming_engine.initialize()
            
            self.logger.info(f"Component {component_name} restarted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to restart component {component_name}: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        with self.status_lock:
            status_dict = {
                'pipeline_active': self.status.pipeline_active,
                'components_initialized': self.status.components_initialized,
                'total_components': self.status.total_components,
                'initialization_complete': self.status.components_initialized == self.status.total_components,
                
                'component_status': {
                    'decision_capture': self.status.decision_capture_ready,
                    'context_processor': self.status.context_processor_ready,
                    'websocket_manager': self.status.websocket_manager_ready,
                    'streaming_engine': self.status.streaming_engine_ready
                },
                
                'health': {
                    'status': self.status.health_status,
                    'last_check': self.status.last_health_check.isoformat() if self.status.last_health_check else None,
                    'component_failures': dict(self.component_failures)
                },
                
                'performance': {
                    'integration_latency_ns': self.status.integration_latency_ns,
                    'total_decisions_processed': self.status.total_decisions_processed,
                    'pipeline_errors': self.status.pipeline_errors
                }
            }
        
        # Add component metrics if available
        if self.decision_capture:
            try:
                status_dict['component_metrics'] = status_dict.get('component_metrics', {})
                status_dict['component_metrics']['decision_capture'] = self.decision_capture.get_metrics()
            except Exception:
                pass
        
        if self.context_processor:
            try:
                status_dict['component_metrics'] = status_dict.get('component_metrics', {})
                status_dict['component_metrics']['context_processor'] = self.context_processor.get_metrics()
            except Exception:
                pass
        
        if self.websocket_manager:
            try:
                status_dict['component_metrics'] = status_dict.get('component_metrics', {})
                status_dict['component_metrics']['websocket_manager'] = self.websocket_manager.get_metrics()
            except Exception:
                pass
        
        if self.streaming_engine:
            try:
                status_dict['component_metrics'] = status_dict.get('component_metrics', {})
                status_dict['component_metrics']['streaming_engine'] = self.streaming_engine.get_metrics()
            except Exception:
                pass
        
        return status_dict
    
    async def graceful_degradation_mode(self) -> None:
        """Enable graceful degradation mode"""
        self.logger.warning("Enabling graceful degradation mode for XAI pipeline")
        
        with self.status_lock:
            self.status.health_status = "degraded"
        
        # Continue operation with available components only
        # Strategic MARL system continues normally without explanations
    
    async def shutdown(self) -> None:
        """Shutdown the XAI Pipeline Integration"""
        try:
            self.logger.info("Shutting down XAI Pipeline Integration...")
            
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await asyncio.wait_for(self.health_check_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            # Shutdown components in reverse order
            shutdown_order = list(reversed(self.component_order))
            
            for component_name in shutdown_order:
                try:
                    if component_name == 'streaming_engine' and self.streaming_engine:
                        await self.streaming_engine.shutdown()
                        self.streaming_engine = None
                        
                    elif component_name == 'context_processor' and self.context_processor:
                        await self.context_processor.shutdown()
                        self.context_processor = None
                        
                    elif component_name == 'decision_capture' and self.decision_capture:
                        await self.decision_capture.shutdown()
                        self.decision_capture = None
                        
                    elif component_name == 'websocket_manager' and self.websocket_manager:
                        await self.websocket_manager.shutdown()
                        self.websocket_manager = None
                    
                    self.logger.info(f"Component {component_name} shutdown successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error shutting down {component_name}: {e}")
            
            # Update status
            with self.status_lock:
                self.status.pipeline_active = False
                self.status.components_initialized = 0
                self.status.health_status = "shutdown"
            
            self.logger.info("XAI Pipeline Integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during XAI Pipeline Integration shutdown: {e}")
            raise


# Integration helper function for easy setup
async def setup_xai_pipeline(kernel, config: Optional[Dict[str, Any]] = None) -> XAIPipelineIntegration:
    """
    Helper function to set up XAI pipeline integration
    
    Args:
        kernel: AlgoSpace kernel
        config: Optional configuration
        
    Returns:
        Initialized XAI pipeline integration
    """
    integration = XAIPipelineIntegration(kernel, config)
    await integration.initialize()
    return integration


# Test function
async def test_xai_integration():
    """Test XAI Pipeline Integration"""
    print("🧪 Testing XAI Pipeline Integration")
    
    # Mock kernel and event bus
    class MockKernel:
        def __init__(self):
            self.event_bus = EventBus()
    
    kernel = MockKernel()
    
    # Test configuration
    config = {
        'health_check_interval_seconds': 5,
        'enable_graceful_degradation': True,
        'websocket_manager': {
            'host': 'localhost',
            'port': 8768,  # Different port for testing
            'authentication': {'enabled': False}
        }
    }
    
    # Initialize integration
    integration = XAIPipelineIntegration(kernel, config)
    await integration.initialize()
    
    # Check status
    status = integration.get_integration_status()
    print(f"\n📊 Integration Status:")
    print(f"  Pipeline active: {status['pipeline_active']}")
    print(f"  Components initialized: {status['components_initialized']}/{status['total_components']}")
    print(f"  Health status: {status['health']['status']}")
    print(f"  Component status: {status['component_status']}")
    
    # Wait for health check
    await asyncio.sleep(1)
    
    # Check metrics
    if 'component_metrics' in status:
        print(f"\n📈 Component Metrics Available:")
        for component, metrics in status.get('component_metrics', {}).items():
            print(f"  {component}: {metrics.get('system_status', {}).get('active', 'unknown')}")
    
    # Shutdown
    await integration.shutdown()
    print("\n✅ XAI Pipeline Integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_xai_integration())