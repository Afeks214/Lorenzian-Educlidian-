"""
Comprehensive test suite for component base functionality.
Tests lifecycle testing, dependency injection, error handling,
and integration with the AlgoSpace kernel system.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from abc import ABC, abstractmethod

from src.core.component_base import ComponentBase


class MockKernel:
    """Mock kernel for testing."""
    
    def __init__(self):
        self.config = {'test_config': 'test_value'}
        self.event_bus = Mock()
        self.event_bus.subscribe = Mock()
        self.event_bus.unsubscribe = Mock()
        self.event_bus.publish = Mock()


class TestComponentBase:
    """Test the ComponentBase abstract class."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
    
    def test_component_base_is_abstract(self):
        """Test that ComponentBase is an abstract class."""
        with pytest.raises(TypeError):
            ComponentBase("test_component", self.mock_kernel)
    
    def test_component_instantiation_fails_without_implementation(self):
        """Test that instantiation fails without implementing abstract methods."""
        class IncompleteComponent(ComponentBase):
            pass
        
        with pytest.raises(TypeError):
            IncompleteComponent("incomplete", self.mock_kernel)
    
    def test_component_requires_all_abstract_methods(self):
        """Test that all abstract methods must be implemented."""
        class PartialComponent(ComponentBase):
            async def initialize(self):
                pass
            
            async def shutdown(self):
                pass
            # Missing get_status method
        
        with pytest.raises(TypeError):
            PartialComponent("partial", self.mock_kernel)


class TestCompleteComponent:
    """Test a complete component implementation."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
        
        # Create a complete component implementation
        class CompleteComponent(ComponentBase):
            def __init__(self, name: str, kernel):
                super().__init__(name, kernel)
                self.initialized = False
                self.shutdown_called = False
                self.status_data = {}
            
            async def initialize(self) -> None:
                self.initialized = True
                self._initialized = True
                self.status_data['initialized'] = True
            
            async def shutdown(self) -> None:
                self.shutdown_called = True
                self._initialized = False
                self.status_data['shutdown'] = True
            
            def get_status(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'initialized': self.initialized,
                    'shutdown_called': self.shutdown_called,
                    'status_data': self.status_data
                }
        
        self.CompleteComponent = CompleteComponent
    
    def test_component_initialization_attributes(self):
        """Test component initialization with correct attributes."""
        component = self.CompleteComponent("test_component", self.mock_kernel)
        
        assert component.name == "test_component"
        assert component.kernel is self.mock_kernel
        assert component.config is self.mock_kernel.config
        assert component.event_bus is self.mock_kernel.event_bus
        assert component._initialized is False
        assert component.is_initialized() is False
    
    @pytest.mark.asyncio
    async def test_component_initialize_method(self):
        """Test component initialization method."""
        component = self.CompleteComponent("test_component", self.mock_kernel)
        
        # Initially not initialized
        assert component.is_initialized() is False
        assert component.initialized is False
        
        # Initialize
        await component.initialize()
        
        # Should be initialized
        assert component.is_initialized() is True
        assert component.initialized is True
        assert component.status_data['initialized'] is True
    
    @pytest.mark.asyncio
    async def test_component_shutdown_method(self):
        """Test component shutdown method."""
        component = self.CompleteComponent("test_component", self.mock_kernel)
        
        # Initialize first
        await component.initialize()
        assert component.is_initialized() is True
        
        # Shutdown
        await component.shutdown()
        
        # Should be shut down
        assert component.is_initialized() is False
        assert component.shutdown_called is True
        assert component.status_data['shutdown'] is True
    
    def test_component_get_status(self):
        """Test component status retrieval."""
        component = self.CompleteComponent("test_component", self.mock_kernel)
        
        status = component.get_status()
        
        assert status['name'] == "test_component"
        assert status['initialized'] is False
        assert status['shutdown_called'] is False
        assert isinstance(status['status_data'], dict)
    
    def test_component_kernel_integration(self):
        """Test component integration with kernel."""
        component = self.CompleteComponent("test_component", self.mock_kernel)
        
        # Should have access to kernel components
        assert component.kernel is self.mock_kernel
        assert component.config == self.mock_kernel.config
        assert component.event_bus is self.mock_kernel.event_bus
        
        # Should be able to access config
        assert component.config['test_config'] == 'test_value'
        
        # Should be able to access event bus
        assert hasattr(component.event_bus, 'subscribe')
        assert hasattr(component.event_bus, 'unsubscribe')
        assert hasattr(component.event_bus, 'publish')


class TestComponentLifecycle:
    """Test component lifecycle management."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
        
        # Create a component that tracks lifecycle events
        class LifecycleComponent(ComponentBase):
            def __init__(self, name: str, kernel):
                super().__init__(name, kernel)
                self.lifecycle_events = []
                self.initialization_error = None
                self.shutdown_error = None
                self.should_fail_init = False
                self.should_fail_shutdown = False
            
            async def initialize(self) -> None:
                self.lifecycle_events.append('initialize_start')
                
                if self.should_fail_init:
                    raise Exception("Initialization failed")
                
                # Simulate initialization work
                await asyncio.sleep(0.01)
                
                self._initialized = True
                self.lifecycle_events.append('initialize_complete')
                
                # Subscribe to some events
                self.event_bus.subscribe('test_event', self._handle_test_event)
            
            async def shutdown(self) -> None:
                self.lifecycle_events.append('shutdown_start')
                
                if self.should_fail_shutdown:
                    raise Exception("Shutdown failed")
                
                # Simulate shutdown work
                await asyncio.sleep(0.01)
                
                # Unsubscribe from events
                self.event_bus.unsubscribe('test_event', self._handle_test_event)
                
                self._initialized = False
                self.lifecycle_events.append('shutdown_complete')
            
            def get_status(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'initialized': self._initialized,
                    'lifecycle_events': self.lifecycle_events,
                    'initialization_error': self.initialization_error,
                    'shutdown_error': self.shutdown_error
                }
            
            def _handle_test_event(self, event_data):
                self.lifecycle_events.append(f'event_received: {event_data}')
        
        self.LifecycleComponent = LifecycleComponent
    
    @pytest.mark.asyncio
    async def test_normal_lifecycle(self):
        """Test normal component lifecycle."""
        component = self.LifecycleComponent("lifecycle_test", self.mock_kernel)
        
        # Initial state
        assert component.is_initialized() is False
        assert len(component.lifecycle_events) == 0
        
        # Initialize
        await component.initialize()
        
        assert component.is_initialized() is True
        assert component.lifecycle_events == ['initialize_start', 'initialize_complete']
        
        # Verify event subscription
        self.mock_kernel.event_bus.subscribe.assert_called_once()
        
        # Shutdown
        await component.shutdown()
        
        assert component.is_initialized() is False
        assert component.lifecycle_events == [
            'initialize_start', 'initialize_complete',
            'shutdown_start', 'shutdown_complete'
        ]
        
        # Verify event unsubscription
        self.mock_kernel.event_bus.unsubscribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test component initialization failure."""
        component = self.LifecycleComponent("lifecycle_test", self.mock_kernel)
        component.should_fail_init = True
        
        # Initialize should fail
        with pytest.raises(Exception, match="Initialization failed"):
            await component.initialize()
        
        # Should remain uninitialized
        assert component.is_initialized() is False
        assert 'initialize_start' in component.lifecycle_events
        assert 'initialize_complete' not in component.lifecycle_events
    
    @pytest.mark.asyncio
    async def test_shutdown_failure(self):
        """Test component shutdown failure."""
        component = self.LifecycleComponent("lifecycle_test", self.mock_kernel)
        
        # Initialize normally
        await component.initialize()
        assert component.is_initialized() is True
        
        # Set shutdown to fail
        component.should_fail_shutdown = True
        
        # Shutdown should fail
        with pytest.raises(Exception, match="Shutdown failed"):
            await component.shutdown()
        
        # Should still be initialized (shutdown failed)
        assert component.is_initialized() is True
        assert 'shutdown_start' in component.lifecycle_events
        assert 'shutdown_complete' not in component.lifecycle_events
    
    @pytest.mark.asyncio
    async def test_double_initialization(self):
        """Test double initialization handling."""
        component = self.LifecycleComponent("lifecycle_test", self.mock_kernel)
        
        # Initialize once
        await component.initialize()
        assert component.is_initialized() is True
        
        # Initialize again
        await component.initialize()
        
        # Should handle gracefully (implementation dependent)
        assert component.is_initialized() is True
        
        # Should have recorded multiple initialization events
        initialize_starts = component.lifecycle_events.count('initialize_start')
        initialize_completes = component.lifecycle_events.count('initialize_complete')
        
        assert initialize_starts == 2
        assert initialize_completes == 2
    
    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self):
        """Test shutdown without initialization."""
        component = self.LifecycleComponent("lifecycle_test", self.mock_kernel)
        
        # Shutdown without initialization
        await component.shutdown()
        
        # Should handle gracefully
        assert component.is_initialized() is False
        assert 'shutdown_start' in component.lifecycle_events
        assert 'shutdown_complete' in component.lifecycle_events


class TestEventBusIntegration:
    """Test component integration with event bus."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
        
        # Create a component that uses event bus
        class EventAwareComponent(ComponentBase):
            def __init__(self, name: str, kernel):
                super().__init__(name, kernel)
                self.received_events = []
                self.published_events = []
                self.subscribed_events = []
            
            async def initialize(self) -> None:
                self._initialized = True
                
                # Subscribe to multiple events
                self.event_bus.subscribe('market_data', self._handle_market_data)
                self.event_bus.subscribe('trade_signal', self._handle_trade_signal)
                self.event_bus.subscribe('risk_alert', self._handle_risk_alert)
                
                self.subscribed_events = ['market_data', 'trade_signal', 'risk_alert']
            
            async def shutdown(self) -> None:
                # Unsubscribe from all events
                for event_type in self.subscribed_events:
                    self.event_bus.unsubscribe(event_type, getattr(self, f'_handle_{event_type}'))
                
                self.subscribed_events = []
                self._initialized = False
            
            def get_status(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'initialized': self._initialized,
                    'received_events': self.received_events,
                    'published_events': self.published_events,
                    'subscribed_events': self.subscribed_events
                }
            
            def _handle_market_data(self, data):
                self.received_events.append(('market_data', data))
            
            def _handle_trade_signal(self, data):
                self.received_events.append(('trade_signal', data))
                
                # Publish a response event
                self.event_bus.publish('trade_response', {'response': 'acknowledged'})
                self.published_events.append(('trade_response', {'response': 'acknowledged'}))
            
            def _handle_risk_alert(self, data):
                self.received_events.append(('risk_alert', data))
                
                # Publish multiple response events
                self.event_bus.publish('alert_acknowledged', {'alert_id': data.get('id')})
                self.event_bus.publish('risk_mitigation', {'action': 'reduce_position'})
                
                self.published_events.append(('alert_acknowledged', {'alert_id': data.get('id')}))
                self.published_events.append(('risk_mitigation', {'action': 'reduce_position'}))
        
        self.EventAwareComponent = EventAwareComponent
    
    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test event subscription during initialization."""
        component = self.EventAwareComponent("event_test", self.mock_kernel)
        
        # Initialize
        await component.initialize()
        
        # Verify subscriptions
        assert len(component.subscribed_events) == 3
        assert 'market_data' in component.subscribed_events
        assert 'trade_signal' in component.subscribed_events
        assert 'risk_alert' in component.subscribed_events
        
        # Verify event bus was called
        assert self.mock_kernel.event_bus.subscribe.call_count == 3
    
    @pytest.mark.asyncio
    async def test_event_unsubscription(self):
        """Test event unsubscription during shutdown."""
        component = self.EventAwareComponent("event_test", self.mock_kernel)
        
        # Initialize and shutdown
        await component.initialize()
        await component.shutdown()
        
        # Verify unsubscriptions
        assert len(component.subscribed_events) == 0
        assert self.mock_kernel.event_bus.unsubscribe.call_count == 3
    
    def test_event_handling(self):
        """Test event handling functionality."""
        component = self.EventAwareComponent("event_test", self.mock_kernel)
        
        # Simulate receiving events
        component._handle_market_data({'price': 100.0, 'volume': 1000})
        component._handle_trade_signal({'action': 'buy', 'quantity': 100})
        component._handle_risk_alert({'id': 'alert_1', 'level': 'high'})
        
        # Verify events were received
        assert len(component.received_events) == 3
        assert component.received_events[0] == ('market_data', {'price': 100.0, 'volume': 1000})
        assert component.received_events[1] == ('trade_signal', {'action': 'buy', 'quantity': 100})
        assert component.received_events[2] == ('risk_alert', {'id': 'alert_1', 'level': 'high'})
        
        # Verify events were published
        assert len(component.published_events) == 3
        assert component.published_events[0] == ('trade_response', {'response': 'acknowledged'})
        assert component.published_events[1] == ('alert_acknowledged', {'alert_id': 'alert_1'})
        assert component.published_events[2] == ('risk_mitigation', {'action': 'reduce_position'})
    
    def test_event_bus_access(self):
        """Test access to event bus functionality."""
        component = self.EventAwareComponent("event_test", self.mock_kernel)
        
        # Should have access to event bus
        assert component.event_bus is self.mock_kernel.event_bus
        
        # Should be able to call event bus methods
        component.event_bus.publish('test_event', {'data': 'test'})
        self.mock_kernel.event_bus.publish.assert_called_with('test_event', {'data': 'test'})
        
        component.event_bus.subscribe('test_event', lambda x: None)
        self.mock_kernel.event_bus.subscribe.assert_called()
        
        component.event_bus.unsubscribe('test_event', lambda x: None)
        self.mock_kernel.event_bus.unsubscribe.assert_called()


class TestDependencyInjection:
    """Test dependency injection functionality."""
    
    def setup_method(self):
        """Setup method for each test."""
        # Create mock kernel with more complex dependencies
        self.mock_kernel = MockKernel()
        self.mock_kernel.database = Mock()
        self.mock_kernel.cache = Mock()
        self.mock_kernel.logger = Mock()
        
        # Create a component that uses injected dependencies
        class DependencyComponent(ComponentBase):
            def __init__(self, name: str, kernel):
                super().__init__(name, kernel)
                
                # Inject dependencies
                self.database = getattr(kernel, 'database', None)
                self.cache = getattr(kernel, 'cache', None)
                self.logger = getattr(kernel, 'logger', None)
                
                self.db_operations = []
                self.cache_operations = []
                self.log_messages = []
            
            async def initialize(self) -> None:
                self._initialized = True
                
                # Use injected dependencies
                if self.database:
                    await self.database.connect()
                    self.db_operations.append('connect')
                
                if self.cache:
                    await self.cache.initialize()
                    self.cache_operations.append('initialize')
                
                if self.logger:
                    self.logger.info(f"Component {self.name} initialized")
                    self.log_messages.append(f"Component {self.name} initialized")
            
            async def shutdown(self) -> None:
                # Use injected dependencies for cleanup
                if self.database:
                    await self.database.disconnect()
                    self.db_operations.append('disconnect')
                
                if self.cache:
                    await self.cache.cleanup()
                    self.cache_operations.append('cleanup')
                
                if self.logger:
                    self.logger.info(f"Component {self.name} shut down")
                    self.log_messages.append(f"Component {self.name} shut down")
                
                self._initialized = False
            
            def get_status(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'initialized': self._initialized,
                    'has_database': self.database is not None,
                    'has_cache': self.cache is not None,
                    'has_logger': self.logger is not None,
                    'db_operations': self.db_operations,
                    'cache_operations': self.cache_operations,
                    'log_messages': self.log_messages
                }
            
            async def perform_business_logic(self):
                """Example business logic using dependencies."""
                if self.database:
                    result = await self.database.query("SELECT * FROM data")
                    self.db_operations.append('query')
                    return result
                return None
            
            def get_cached_data(self, key: str):
                """Example cache usage."""
                if self.cache:
                    result = self.cache.get(key)
                    self.cache_operations.append(f'get:{key}')
                    return result
                return None
        
        self.DependencyComponent = DependencyComponent
    
    def test_dependency_injection(self):
        """Test that dependencies are properly injected."""
        # Setup mock methods
        self.mock_kernel.database.connect = AsyncMock()
        self.mock_kernel.database.disconnect = AsyncMock()
        self.mock_kernel.database.query = AsyncMock(return_value=[{'id': 1, 'data': 'test'}])
        
        self.mock_kernel.cache.initialize = AsyncMock()
        self.mock_kernel.cache.cleanup = AsyncMock()
        self.mock_kernel.cache.get = Mock(return_value='cached_value')
        
        self.mock_kernel.logger.info = Mock()
        
        component = self.DependencyComponent("dependency_test", self.mock_kernel)
        
        # Verify dependencies were injected
        assert component.database is self.mock_kernel.database
        assert component.cache is self.mock_kernel.cache
        assert component.logger is self.mock_kernel.logger
        
        status = component.get_status()
        assert status['has_database'] is True
        assert status['has_cache'] is True
        assert status['has_logger'] is True
    
    @pytest.mark.asyncio
    async def test_dependency_usage_in_lifecycle(self):
        """Test dependency usage during component lifecycle."""
        # Setup mock methods
        self.mock_kernel.database.connect = AsyncMock()
        self.mock_kernel.database.disconnect = AsyncMock()
        self.mock_kernel.cache.initialize = AsyncMock()
        self.mock_kernel.cache.cleanup = AsyncMock()
        self.mock_kernel.logger.info = Mock()
        
        component = self.DependencyComponent("dependency_test", self.mock_kernel)
        
        # Initialize
        await component.initialize()
        
        # Verify dependencies were used
        self.mock_kernel.database.connect.assert_called_once()
        self.mock_kernel.cache.initialize.assert_called_once()
        self.mock_kernel.logger.info.assert_called_once()
        
        assert 'connect' in component.db_operations
        assert 'initialize' in component.cache_operations
        assert 'Component dependency_test initialized' in component.log_messages
        
        # Shutdown
        await component.shutdown()
        
        # Verify cleanup was performed
        self.mock_kernel.database.disconnect.assert_called_once()
        self.mock_kernel.cache.cleanup.assert_called_once()
        
        assert 'disconnect' in component.db_operations
        assert 'cleanup' in component.cache_operations
        assert 'Component dependency_test shut down' in component.log_messages
    
    @pytest.mark.asyncio
    async def test_dependency_usage_in_business_logic(self):
        """Test dependency usage in business logic."""
        # Setup mock methods
        self.mock_kernel.database.query = AsyncMock(return_value=[{'id': 1, 'data': 'test'}])
        self.mock_kernel.cache.get = Mock(return_value='cached_value')
        
        component = self.DependencyComponent("dependency_test", self.mock_kernel)
        
        # Test business logic
        result = await component.perform_business_logic()
        
        # Verify database was used
        self.mock_kernel.database.query.assert_called_once_with("SELECT * FROM data")
        assert result == [{'id': 1, 'data': 'test'}]
        assert 'query' in component.db_operations
        
        # Test cache usage
        cached_result = component.get_cached_data('test_key')
        
        # Verify cache was used
        self.mock_kernel.cache.get.assert_called_once_with('test_key')
        assert cached_result == 'cached_value'
        assert 'get:test_key' in component.cache_operations
    
    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        # Create kernel without some dependencies
        minimal_kernel = MockKernel()
        # No database, cache, or logger
        
        component = self.DependencyComponent("dependency_test", minimal_kernel)
        
        # Should handle missing dependencies gracefully
        assert component.database is None
        assert component.cache is None
        assert component.logger is None
        
        status = component.get_status()
        assert status['has_database'] is False
        assert status['has_cache'] is False
        assert status['has_logger'] is False
    
    @pytest.mark.asyncio
    async def test_partial_dependencies(self):
        """Test handling of partial dependencies."""
        # Create kernel with only some dependencies
        partial_kernel = MockKernel()
        partial_kernel.database = Mock()
        partial_kernel.database.connect = AsyncMock()
        partial_kernel.database.disconnect = AsyncMock()
        # No cache or logger
        
        component = self.DependencyComponent("dependency_test", partial_kernel)
        
        # Should handle partial dependencies
        assert component.database is not None
        assert component.cache is None
        assert component.logger is None
        
        # Initialize should work with partial dependencies
        await component.initialize()
        
        # Only database should be used
        partial_kernel.database.connect.assert_called_once()
        assert 'connect' in component.db_operations
        assert len(component.cache_operations) == 0
        assert len(component.log_messages) == 0


class TestErrorHandling:
    """Test error handling in components."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
        
        # Create a component that can simulate various errors
        class ErrorProneComponent(ComponentBase):
            def __init__(self, name: str, kernel):
                super().__init__(name, kernel)
                self.initialization_errors = []
                self.shutdown_errors = []
                self.runtime_errors = []
                
                # Error simulation flags
                self.should_fail_init = False
                self.should_fail_shutdown = False
                self.should_fail_status = False
                
                self.init_error_type = Exception
                self.shutdown_error_type = Exception
                self.status_error_type = Exception
            
            async def initialize(self) -> None:
                try:
                    if self.should_fail_init:
                        raise self.init_error_type("Initialization failed")
                    
                    # Simulate some initialization work
                    await asyncio.sleep(0.001)
                    
                    self._initialized = True
                    
                except Exception as e:
                    self.initialization_errors.append(str(e))
                    raise
            
            async def shutdown(self) -> None:
                try:
                    if self.should_fail_shutdown:
                        raise self.shutdown_error_type("Shutdown failed")
                    
                    # Simulate some shutdown work
                    await asyncio.sleep(0.001)
                    
                    self._initialized = False
                    
                except Exception as e:
                    self.shutdown_errors.append(str(e))
                    raise
            
            def get_status(self) -> Dict[str, Any]:
                try:
                    if self.should_fail_status:
                        raise self.status_error_type("Status retrieval failed")
                    
                    return {
                        'name': self.name,
                        'initialized': self._initialized,
                        'initialization_errors': self.initialization_errors,
                        'shutdown_errors': self.shutdown_errors,
                        'runtime_errors': self.runtime_errors
                    }
                    
                except Exception as e:
                    self.runtime_errors.append(str(e))
                    raise
            
            async def do_risky_operation(self):
                """Simulate a risky operation that might fail."""
                import random
                if random.random() < 0.3:  # 30% chance of failure
                    raise RuntimeError("Risky operation failed")
                return "Success"
        
        self.ErrorProneComponent = ErrorProneComponent
    
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        component.should_fail_init = True
        
        # Initialization should fail
        with pytest.raises(Exception, match="Initialization failed"):
            await component.initialize()
        
        # Component should remain uninitialized
        assert component.is_initialized() is False
        assert "Initialization failed" in component.initialization_errors
    
    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self):
        """Test error handling during shutdown."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        
        # Initialize normally
        await component.initialize()
        assert component.is_initialized() is True
        
        # Set shutdown to fail
        component.should_fail_shutdown = True
        
        # Shutdown should fail
        with pytest.raises(Exception, match="Shutdown failed"):
            await component.shutdown()
        
        # Component should still be initialized (shutdown failed)
        assert component.is_initialized() is True
        assert "Shutdown failed" in component.shutdown_errors
    
    def test_status_error_handling(self):
        """Test error handling in status retrieval."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        component.should_fail_status = True
        
        # Status retrieval should fail
        with pytest.raises(Exception, match="Status retrieval failed"):
            component.get_status()
        
        # Error should be recorded
        assert "Status retrieval failed" in component.runtime_errors
    
    @pytest.mark.asyncio
    async def test_different_error_types(self):
        """Test handling of different error types."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        
        # Test ValueError in initialization
        component.should_fail_init = True
        component.init_error_type = ValueError
        
        with pytest.raises(ValueError):
            await component.initialize()
        
        # Reset and test RuntimeError in shutdown
        component.should_fail_init = False
        component.should_fail_shutdown = True
        component.shutdown_error_type = RuntimeError
        
        await component.initialize()
        
        with pytest.raises(RuntimeError):
            await component.shutdown()
        
        # Test TypeError in status
        component.should_fail_status = True
        component.status_error_type = TypeError
        
        with pytest.raises(TypeError):
            component.get_status()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery scenarios."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        
        # Fail initialization first
        component.should_fail_init = True
        
        with pytest.raises(Exception):
            await component.initialize()
        
        assert component.is_initialized() is False
        
        # Recover from initialization error
        component.should_fail_init = False
        
        # Should be able to initialize successfully now
        await component.initialize()
        assert component.is_initialized() is True
        
        # Test shutdown recovery
        component.should_fail_shutdown = True
        
        with pytest.raises(Exception):
            await component.shutdown()
        
        # Still initialized due to shutdown failure
        assert component.is_initialized() is True
        
        # Recover from shutdown error
        component.should_fail_shutdown = False
        
        # Should be able to shutdown successfully now
        await component.shutdown()
        assert component.is_initialized() is False
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling."""
        component = self.ErrorProneComponent("error_test", self.mock_kernel)
        
        # Test multiple async operations with some failing
        results = []
        
        for i in range(10):
            try:
                result = await component.do_risky_operation()
                results.append(('success', result))
            except RuntimeError as e:
                results.append(('error', str(e)))
        
        # Should have a mix of successes and errors
        successes = [r for r in results if r[0] == 'success']
        errors = [r for r in results if r[0] == 'error']
        
        assert len(successes) > 0  # Should have some successes
        assert len(errors) > 0     # Should have some errors
        assert len(successes) + len(errors) == 10  # Total should be 10


class TestComponentIntegration:
    """Test integration scenarios with multiple components."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.mock_kernel = MockKernel()
        
        # Create multiple component types
        class DataProcessorComponent(ComponentBase):
            async def initialize(self):
                self._initialized = True
                self.processed_data = []
            
            async def shutdown(self):
                self._initialized = False
            
            def get_status(self):
                return {'name': self.name, 'processed_count': len(self.processed_data)}
            
            def process_data(self, data):
                self.processed_data.append(f"processed_{data}")
                return f"processed_{data}"
        
        class NotificationComponent(ComponentBase):
            async def initialize(self):
                self._initialized = True
                self.notifications = []
            
            async def shutdown(self):
                self._initialized = False
            
            def get_status(self):
                return {'name': self.name, 'notification_count': len(self.notifications)}
            
            def send_notification(self, message):
                self.notifications.append(message)
        
        self.DataProcessorComponent = DataProcessorComponent
        self.NotificationComponent = NotificationComponent
    
    @pytest.mark.asyncio
    async def test_multiple_component_lifecycle(self):
        """Test lifecycle management with multiple components."""
        # Create multiple components
        processor = self.DataProcessorComponent("processor", self.mock_kernel)
        notifier = self.NotificationComponent("notifier", self.mock_kernel)
        
        components = [processor, notifier]
        
        # Initialize all components
        for component in components:
            await component.initialize()
            assert component.is_initialized() is True
        
        # Use components
        processor.process_data("test_data")
        notifier.send_notification("Processing complete")
        
        # Check statuses
        processor_status = processor.get_status()
        notifier_status = notifier.get_status()
        
        assert processor_status['processed_count'] == 1
        assert notifier_status['notification_count'] == 1
        
        # Shutdown all components
        for component in components:
            await component.shutdown()
            assert component.is_initialized() is False
    
    @pytest.mark.asyncio
    async def test_component_interaction(self):
        """Test interaction between components."""
        # Create interacting components
        processor = self.DataProcessorComponent("processor", self.mock_kernel)
        notifier = self.NotificationComponent("notifier", self.mock_kernel)
        
        # Initialize
        await processor.initialize()
        await notifier.initialize()
        
        # Simulate component interaction
        for i in range(5):
            result = processor.process_data(f"data_{i}")
            notifier.send_notification(f"Processed: {result}")
        
        # Verify interaction results
        processor_status = processor.get_status()
        notifier_status = notifier.get_status()
        
        assert processor_status['processed_count'] == 5
        assert notifier_status['notification_count'] == 5
        
        # Verify data consistency
        assert len(processor.processed_data) == 5
        assert len(notifier.notifications) == 5
        
        # Shutdown
        await processor.shutdown()
        await notifier.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_dependency_chain(self):
        """Test components with dependency chains."""
        # Create component that depends on others
        class ManagerComponent(ComponentBase):
            def __init__(self, name, kernel, processor, notifier):
                super().__init__(name, kernel)
                self.processor = processor
                self.notifier = notifier
                self.operations = []
            
            async def initialize(self):
                # Ensure dependencies are initialized
                if not self.processor.is_initialized():
                    await self.processor.initialize()
                if not self.notifier.is_initialized():
                    await self.notifier.initialize()
                
                self._initialized = True
            
            async def shutdown(self):
                # Shutdown in reverse order
                await self.notifier.shutdown()
                await self.processor.shutdown()
                self._initialized = False
            
            def get_status(self):
                return {
                    'name': self.name,
                    'operations': len(self.operations),
                    'processor_status': self.processor.get_status(),
                    'notifier_status': self.notifier.get_status()
                }
            
            def perform_operation(self, data):
                result = self.processor.process_data(data)
                self.notifier.send_notification(f"Operation completed: {result}")
                self.operations.append((data, result))
                return result
        
        # Create component chain
        processor = self.DataProcessorComponent("processor", self.mock_kernel)
        notifier = self.NotificationComponent("notifier", self.mock_kernel)
        manager = ManagerComponent("manager", self.mock_kernel, processor, notifier)
        
        # Initialize manager (should initialize dependencies)
        await manager.initialize()
        
        # Verify all components are initialized
        assert manager.is_initialized() is True
        assert processor.is_initialized() is True
        assert notifier.is_initialized() is True
        
        # Perform operations
        results = []
        for i in range(3):
            result = manager.perform_operation(f"data_{i}")
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        assert all(r.startswith("processed_") for r in results)
        
        # Verify status
        status = manager.get_status()
        assert status['operations'] == 3
        assert status['processor_status']['processed_count'] == 3
        assert status['notifier_status']['notification_count'] == 3
        
        # Shutdown manager (should shutdown dependencies)
        await manager.shutdown()
        
        # Verify all components are shut down
        assert manager.is_initialized() is False
        assert processor.is_initialized() is False
        assert notifier.is_initialized() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])