"""
Comprehensive WebSocket handlers testing with security focus.
Tests connection management, message handling, error recovery, and stress testing.
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import weakref
from dataclasses import asdict

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import concurrent.futures

from src.xai.api.websocket_handlers import (
    MessageType,
    SubscriptionType,
    WebSocketConnection,
    Subscription,
    WebSocketMessage,
    ConnectionManager,
    SubscriptionManager,
    WebSocketConnectionManager
)
from src.core.event_bus import EventBus
from src.core.events import EventType, Event

# Test fixtures
@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    websocket = AsyncMock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    websocket.client_state = WebSocketState.CONNECTED
    websocket.client = Mock()
    websocket.client.host = "127.0.0.1"
    websocket.headers = {"user-agent": "test-client"}
    return websocket

@pytest.fixture
def mock_event_bus():
    """Mock EventBus for testing."""
    event_bus = Mock(spec=EventBus)
    event_bus.subscribe = Mock()
    event_bus.unsubscribe = Mock()
    event_bus.publish = Mock()
    return event_bus

@pytest.fixture
def sample_websocket_message():
    """Sample WebSocket message for testing."""
    return WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType.EXPLANATION,
        timestamp=datetime.now(),
        data={"explanation": "test explanation", "symbol": "AAPL", "agent": "MLMI"},
        correlation_id="test_correlation"
    )

@pytest.fixture
def sample_subscription():
    """Sample subscription for testing."""
    return Subscription(
        connection_id="test_connection",
        subscription_type=SubscriptionType.ALL_EXPLANATIONS,
        filters={"symbol": "AAPL"},
        created_at=datetime.now()
    )

@pytest.fixture
def connection_manager():
    """Connection manager instance for testing."""
    return ConnectionManager()

@pytest.fixture
def subscription_manager():
    """Subscription manager instance for testing."""
    return SubscriptionManager()

@pytest.fixture
def websocket_connection_manager():
    """WebSocket connection manager instance for testing."""
    return WebSocketConnectionManager()

class TestWebSocketConnection:
    """Test WebSocketConnection dataclass."""
    
    def test_websocket_connection_creation(self, mock_websocket):
        """Test WebSocket connection creation."""
        connection = WebSocketConnection(
            id="test_connection",
            websocket=mock_websocket,
            user_id="test_user",
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            subscriptions=set(["test_subscription"]),
            connection_info={"client_host": "127.0.0.1"},
            message_count=0,
            bytes_sent=0,
            bytes_received=0
        )
        
        assert connection.id == "test_connection"
        assert connection.websocket == mock_websocket
        assert connection.user_id == "test_user"
        assert isinstance(connection.connected_at, datetime)
        assert isinstance(connection.last_activity, datetime)
        assert connection.subscriptions == {"test_subscription"}
        assert connection.connection_info == {"client_host": "127.0.0.1"}
        assert connection.message_count == 0
        assert connection.bytes_sent == 0
        assert connection.bytes_received == 0

class TestWebSocketMessage:
    """Test WebSocketMessage dataclass."""
    
    def test_websocket_message_creation(self):
        """Test WebSocket message creation."""
        message = WebSocketMessage(
            id="test_id",
            type=MessageType.EXPLANATION,
            timestamp=datetime.now(),
            data={"test": "data"},
            correlation_id="test_correlation",
            target_connection="test_connection"
        )
        
        assert message.id == "test_id"
        assert message.type == MessageType.EXPLANATION
        assert isinstance(message.timestamp, datetime)
        assert message.data == {"test": "data"}
        assert message.correlation_id == "test_correlation"
        assert message.target_connection == "test_connection"
    
    def test_websocket_message_optional_fields(self):
        """Test WebSocket message with optional fields."""
        message = WebSocketMessage(
            id="test_id",
            type=MessageType.PING,
            timestamp=datetime.now(),
            data={"ping": "pong"}
        )
        
        assert message.correlation_id is None
        assert message.target_connection is None

class TestConnectionManager:
    """Test ConnectionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket connection."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection",
            user_id="test_user",
            connection_info={"client_host": "127.0.0.1"}
        )
        
        assert connection_id == "test_connection"
        assert "test_connection" in connection_manager.connections
        assert mock_websocket in connection_manager.connection_lookup
        mock_websocket.accept.assert_called_once()
        
        connection = connection_manager.connections["test_connection"]
        assert connection.user_id == "test_user"
        assert connection.connection_info == {"client_host": "127.0.0.1"}
    
    @pytest.mark.asyncio
    async def test_connect_websocket_without_user_id(self, connection_manager, mock_websocket):
        """Test WebSocket connection without user ID."""
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        )
        
        assert connection_id == "test_connection"
        connection = connection_manager.connections["test_connection"]
        assert connection.user_id is None
        assert connection.connection_info == {}
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection."""
        # First connect
        await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection",
            user_id="test_user"
        )
        
        # Then disconnect
        await connection_manager.disconnect("test_connection")
        
        assert "test_connection" not in connection_manager.connections
        assert mock_websocket not in connection_manager.connection_lookup
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self, connection_manager):
        """Test disconnecting non-existent connection."""
        # Should not raise exception
        await connection_manager.disconnect("nonexistent_connection")
        assert len(connection_manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket_error(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection with error."""
        mock_websocket.close.side_effect = Exception("Close error")
        
        await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        )
        
        # Should handle error gracefully
        await connection_manager.disconnect("test_connection")
        
        assert "test_connection" not in connection_manager.connections
    
    def test_get_connection(self, connection_manager, mock_websocket):
        """Test getting connection by ID."""
        # Connect first
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection",
            user_id="test_user"
        ))
        
        connection = connection_manager.get_connection("test_connection")
        assert connection is not None
        assert connection.id == "test_connection"
        assert connection.user_id == "test_user"
        
        # Test non-existent connection
        connection = connection_manager.get_connection("nonexistent")
        assert connection is None
    
    def test_get_connection_by_websocket(self, connection_manager, mock_websocket):
        """Test getting connection by WebSocket instance."""
        # Connect first
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        ))
        
        connection = connection_manager.get_connection_by_websocket(mock_websocket)
        assert connection is not None
        assert connection.id == "test_connection"
        
        # Test non-existent WebSocket
        other_websocket = Mock()
        connection = connection_manager.get_connection_by_websocket(other_websocket)
        assert connection is None
    
    def test_get_all_connections(self, connection_manager, mock_websocket):
        """Test getting all connections."""
        # Initially empty
        connections = connection_manager.get_all_connections()
        assert len(connections) == 0
        
        # Connect multiple WebSockets
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection_1"
        ))
        
        mock_websocket2 = AsyncMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.client_state = WebSocketState.CONNECTED
        mock_websocket2.client = Mock()
        mock_websocket2.client.host = "127.0.0.1"
        mock_websocket2.headers = {"user-agent": "test-client"}
        
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket2,
            connection_id="test_connection_2"
        ))
        
        connections = connection_manager.get_all_connections()
        assert len(connections) == 2
        assert all(isinstance(conn, WebSocketConnection) for conn in connections)
    
    def test_get_connections_by_user(self, connection_manager, mock_websocket):
        """Test getting connections by user ID."""
        # Connect multiple WebSockets for different users
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection_1",
            user_id="user1"
        ))
        
        mock_websocket2 = AsyncMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.client_state = WebSocketState.CONNECTED
        mock_websocket2.client = Mock()
        mock_websocket2.client.host = "127.0.0.1"
        mock_websocket2.headers = {"user-agent": "test-client"}
        
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket2,
            connection_id="test_connection_2",
            user_id="user2"
        ))
        
        mock_websocket3 = AsyncMock(spec=WebSocket)
        mock_websocket3.accept = AsyncMock()
        mock_websocket3.client_state = WebSocketState.CONNECTED
        mock_websocket3.client = Mock()
        mock_websocket3.client.host = "127.0.0.1"
        mock_websocket3.headers = {"user-agent": "test-client"}
        
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket3,
            connection_id="test_connection_3",
            user_id="user1"
        ))
        
        # Get connections for user1
        user1_connections = connection_manager.get_connections_by_user("user1")
        assert len(user1_connections) == 2
        assert all(conn.user_id == "user1" for conn in user1_connections)
        
        # Get connections for user2
        user2_connections = connection_manager.get_connections_by_user("user2")
        assert len(user2_connections) == 1
        assert user2_connections[0].user_id == "user2"
        
        # Get connections for non-existent user
        nonexistent_connections = connection_manager.get_connections_by_user("nonexistent")
        assert len(nonexistent_connections) == 0
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, connection_manager, mock_websocket, sample_websocket_message):
        """Test successful message sending."""
        await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        )
        
        success = await connection_manager.send_message("test_connection", sample_websocket_message)
        
        assert success is True
        mock_websocket.send_json.assert_called_once()
        
        # Check message format
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["id"] == sample_websocket_message.id
        assert call_args["type"] == sample_websocket_message.type.value
        assert call_args["data"] == sample_websocket_message.data
        assert call_args["correlation_id"] == sample_websocket_message.correlation_id
        
        # Check connection stats updated
        connection = connection_manager.connections["test_connection"]
        assert connection.message_count == 1
        assert connection.bytes_sent > 0
    
    @pytest.mark.asyncio
    async def test_send_message_nonexistent_connection(self, connection_manager, sample_websocket_message):
        """Test sending message to non-existent connection."""
        success = await connection_manager.send_message("nonexistent", sample_websocket_message)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_send_message_disconnected_websocket(self, connection_manager, mock_websocket, sample_websocket_message):
        """Test sending message to disconnected WebSocket."""
        await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        )
        
        # Simulate disconnected WebSocket
        mock_websocket.client_state = WebSocketState.DISCONNECTED
        
        success = await connection_manager.send_message("test_connection", sample_websocket_message)
        
        assert success is False
        # Connection should be cleaned up
        assert "test_connection" not in connection_manager.connections
    
    @pytest.mark.asyncio
    async def test_send_message_error(self, connection_manager, mock_websocket, sample_websocket_message):
        """Test sending message with error."""
        await connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        )
        
        mock_websocket.send_json.side_effect = Exception("Send error")
        
        success = await connection_manager.send_message("test_connection", sample_websocket_message)
        
        assert success is False
        # Connection should be cleaned up
        assert "test_connection" not in connection_manager.connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message_success(self, connection_manager, sample_websocket_message):
        """Test successful message broadcasting."""
        # Connect multiple WebSockets
        mock_websockets = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = "127.0.0.1"
            mock_ws.headers = {"user-agent": "test-client"}
            mock_websockets.append(mock_ws)
            
            await connection_manager.connect(
                websocket=mock_ws,
                connection_id=f"test_connection_{i}"
            )
        
        success_count = await connection_manager.broadcast_message(sample_websocket_message)
        
        assert success_count == 3
        for mock_ws in mock_websockets:
            mock_ws.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_message_with_filter(self, connection_manager, sample_websocket_message):
        """Test message broadcasting with filter."""
        # Connect multiple WebSockets with different users
        mock_websockets = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = "127.0.0.1"
            mock_ws.headers = {"user-agent": "test-client"}
            mock_websockets.append(mock_ws)
            
            await connection_manager.connect(
                websocket=mock_ws,
                connection_id=f"test_connection_{i}",
                user_id=f"user_{i % 2}"  # Two different users
            )
        
        # Filter for user_0 only
        def filter_func(conn):
            return conn.user_id == "user_0"
        
        success_count = await connection_manager.broadcast_message(
            sample_websocket_message,
            filter_func=filter_func
        )
        
        assert success_count == 2  # Two connections have user_0
        # Only user_0 connections should receive message
        mock_websockets[0].send_json.assert_called_once()
        mock_websockets[1].send_json.assert_not_called()
        mock_websockets[2].send_json.assert_called_once()
    
    def test_cleanup_stale_connections(self, connection_manager, mock_websocket):
        """Test cleanup of stale connections."""
        # Connect WebSocket
        asyncio.run(connection_manager.connect(
            websocket=mock_websocket,
            connection_id="test_connection"
        ))
        
        # Simulate stale connection by setting old last_activity
        connection = connection_manager.connections["test_connection"]
        connection.last_activity = datetime.now() - timedelta(minutes=35)
        
        # Cleanup with 30 minute timeout
        cleaned_count = connection_manager.cleanup_stale_connections(timeout_minutes=30)
        
        assert cleaned_count == 1
        # Note: cleanup_stale_connections creates async tasks, so connection may still be there
        # In real implementation, we'd need to wait for the task to complete

class TestSubscriptionManager:
    """Test SubscriptionManager functionality."""
    
    def test_subscribe_success(self, subscription_manager):
        """Test successful subscription."""
        subscription_id = subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS,
            filters={"symbol": "AAPL"}
        )
        
        assert subscription_id is not None
        assert subscription_id.startswith("test_connection_all_explanations_")
        
        # Check subscription was added
        assert SubscriptionType.ALL_EXPLANATIONS.value in subscription_manager.subscriptions
        assert len(subscription_manager.subscriptions[SubscriptionType.ALL_EXPLANATIONS.value]) == 1
        
        # Check connection subscriptions
        assert "test_connection" in subscription_manager.connection_subscriptions
        assert len(subscription_manager.connection_subscriptions["test_connection"]) == 1
    
    def test_subscribe_without_filters(self, subscription_manager):
        """Test subscription without filters."""
        subscription_id = subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        assert subscription_id is not None
        
        # Check subscription was added with empty filters
        subscriptions = subscription_manager.subscriptions[SubscriptionType.DECISION_NOTIFICATIONS.value]
        assert len(subscriptions) == 1
        assert subscriptions[0].filters == {}
    
    def test_subscribe_multiple_types(self, subscription_manager):
        """Test multiple subscription types for same connection."""
        # Subscribe to different types
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        # Check both subscriptions exist
        assert len(subscription_manager.subscriptions[SubscriptionType.ALL_EXPLANATIONS.value]) == 1
        assert len(subscription_manager.subscriptions[SubscriptionType.DECISION_NOTIFICATIONS.value]) == 1
        
        # Check connection has both subscriptions
        assert len(subscription_manager.connection_subscriptions["test_connection"]) == 2
    
    def test_unsubscribe_specific_type(self, subscription_manager):
        """Test unsubscribing from specific subscription type."""
        # Subscribe to multiple types
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        # Unsubscribe from one type
        removed_count = subscription_manager.unsubscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        assert removed_count == 1
        
        # Check specific subscription was removed
        assert len(subscription_manager.subscriptions[SubscriptionType.ALL_EXPLANATIONS.value]) == 0
        assert len(subscription_manager.subscriptions[SubscriptionType.DECISION_NOTIFICATIONS.value]) == 1
        
        # Check connection still has one subscription
        assert len(subscription_manager.connection_subscriptions["test_connection"]) == 1
    
    def test_unsubscribe_all_types(self, subscription_manager):
        """Test unsubscribing from all subscription types."""
        # Subscribe to multiple types
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        # Unsubscribe from all types (None parameter)
        removed_count = subscription_manager.unsubscribe(
            connection_id="test_connection",
            subscription_type=None
        )
        
        assert removed_count == 2
        
        # Check all subscriptions were removed
        assert len(subscription_manager.subscriptions[SubscriptionType.ALL_EXPLANATIONS.value]) == 0
        assert len(subscription_manager.subscriptions[SubscriptionType.DECISION_NOTIFICATIONS.value]) == 0
        
        # Check connection has no subscriptions
        assert len(subscription_manager.connection_subscriptions["test_connection"]) == 0
    
    def test_unsubscribe_nonexistent_connection(self, subscription_manager):
        """Test unsubscribing from non-existent connection."""
        removed_count = subscription_manager.unsubscribe(
            connection_id="nonexistent_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        assert removed_count == 0
    
    def test_get_subscribers_all_explanations(self, subscription_manager):
        """Test getting subscribers for all explanations."""
        # Subscribe multiple connections
        subscription_manager.subscribe(
            connection_id="connection_1",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscription_manager.subscribe(
            connection_id="connection_2",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscribers = subscription_manager.get_subscribers(
            SubscriptionType.ALL_EXPLANATIONS
        )
        
        assert len(subscribers) == 2
        assert "connection_1" in subscribers
        assert "connection_2" in subscribers
    
    def test_get_subscribers_with_filter_match(self, subscription_manager):
        """Test getting subscribers with matching filter."""
        # Subscribe with symbol filter
        subscription_manager.subscribe(
            connection_id="connection_1",
            subscription_type=SubscriptionType.SYMBOL_EXPLANATIONS,
            filters={"symbol": "AAPL"}
        )
        
        subscription_manager.subscribe(
            connection_id="connection_2",
            subscription_type=SubscriptionType.SYMBOL_EXPLANATIONS,
            filters={"symbol": "MSFT"}
        )
        
        # Get subscribers for AAPL
        subscribers = subscription_manager.get_subscribers(
            SubscriptionType.SYMBOL_EXPLANATIONS,
            message_data={"symbol": "AAPL"}
        )
        
        assert len(subscribers) == 1
        assert "connection_1" in subscribers
        assert "connection_2" not in subscribers
    
    def test_get_subscribers_with_filter_no_match(self, subscription_manager):
        """Test getting subscribers with no matching filter."""
        # Subscribe with symbol filter
        subscription_manager.subscribe(
            connection_id="connection_1",
            subscription_type=SubscriptionType.SYMBOL_EXPLANATIONS,
            filters={"symbol": "AAPL"}
        )
        
        # Get subscribers for different symbol
        subscribers = subscription_manager.get_subscribers(
            SubscriptionType.SYMBOL_EXPLANATIONS,
            message_data={"symbol": "MSFT"}
        )
        
        assert len(subscribers) == 0
    
    def test_get_subscribers_no_message_data(self, subscription_manager):
        """Test getting subscribers without message data."""
        # Subscribe with filters
        subscription_manager.subscribe(
            connection_id="connection_1",
            subscription_type=SubscriptionType.SYMBOL_EXPLANATIONS,
            filters={"symbol": "AAPL"}
        )
        
        # Get subscribers without message data (should match all)
        subscribers = subscription_manager.get_subscribers(
            SubscriptionType.SYMBOL_EXPLANATIONS,
            message_data=None
        )
        
        assert len(subscribers) == 1
        assert "connection_1" in subscribers
    
    def test_get_subscribers_empty_filters(self, subscription_manager):
        """Test getting subscribers with empty filters."""
        # Subscribe without filters
        subscription_manager.subscribe(
            connection_id="connection_1",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS,
            filters={}
        )
        
        # Get subscribers with any message data
        subscribers = subscription_manager.get_subscribers(
            SubscriptionType.ALL_EXPLANATIONS,
            message_data={"symbol": "AAPL", "agent": "MLMI"}
        )
        
        assert len(subscribers) == 1
        assert "connection_1" in subscribers
    
    def test_cleanup_connection_subscriptions(self, subscription_manager):
        """Test cleanup of connection subscriptions."""
        # Subscribe to multiple types
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        subscription_manager.subscribe(
            connection_id="test_connection",
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        # Cleanup connection subscriptions
        removed_count = subscription_manager.cleanup_connection_subscriptions("test_connection")
        
        assert removed_count == 2
        
        # Check all subscriptions were removed
        assert len(subscription_manager.subscriptions[SubscriptionType.ALL_EXPLANATIONS.value]) == 0
        assert len(subscription_manager.subscriptions[SubscriptionType.DECISION_NOTIFICATIONS.value]) == 0
        assert len(subscription_manager.connection_subscriptions["test_connection"]) == 0

class TestWebSocketConnectionManager:
    """Test WebSocketConnectionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, websocket_connection_manager):
        """Test WebSocket connection manager initialization."""
        # Test with default config
        assert websocket_connection_manager.config is not None
        assert websocket_connection_manager.connection_manager is not None
        assert websocket_connection_manager.subscription_manager is not None
        assert websocket_connection_manager.metrics is not None
        
        # Test metrics initialization
        metrics = websocket_connection_manager.metrics
        assert metrics['total_connections'] == 0
        assert metrics['active_connections'] == 0
        assert metrics['messages_sent'] == 0
        assert metrics['messages_received'] == 0
    
    @pytest.mark.asyncio
    async def test_initialization_with_custom_config(self):
        """Test WebSocket connection manager with custom config."""
        custom_config = {
            'heartbeat_interval_seconds': 60,
            'cleanup_interval_minutes': 10,
            'max_connections_per_user': 10
        }
        
        manager = WebSocketConnectionManager(config=custom_config)
        
        assert manager.config['heartbeat_interval_seconds'] == 60
        assert manager.config['cleanup_interval_minutes'] == 10
        assert manager.config['max_connections_per_user'] == 10
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, websocket_connection_manager):
        """Test successful initialization."""
        with patch('src.xai.api.websocket_handlers.EventBus') as mock_event_bus_class:
            mock_event_bus = Mock()
            mock_event_bus_class.return_value = mock_event_bus
            
            await websocket_connection_manager.initialize()
            
            assert websocket_connection_manager.event_bus is not None
            assert websocket_connection_manager._cleanup_task is not None
            assert websocket_connection_manager._heartbeat_task is not None
            
            # Check event subscriptions
            assert mock_event_bus.subscribe.call_count == 3
    
    @pytest.mark.asyncio
    async def test_connect_success(self, websocket_connection_manager, mock_websocket):
        """Test successful WebSocket connection."""
        connection_id = await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation",
            user_id="test_user"
        )
        
        assert connection_id is not None
        assert connection_id.startswith("ws_")
        assert len(websocket_connection_manager.connection_manager.connections) == 1
        assert websocket_connection_manager.metrics['total_connections'] == 1
        assert websocket_connection_manager.metrics['active_connections'] == 1
        
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_without_user_id(self, websocket_connection_manager, mock_websocket):
        """Test WebSocket connection without user ID."""
        connection_id = await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        assert connection_id is not None
        assert len(websocket_connection_manager.connection_manager.connections) == 1
    
    @pytest.mark.asyncio
    async def test_connect_max_connections_exceeded(self, websocket_connection_manager, mock_websocket):
        """Test connection limit exceeded."""
        # Set low connection limit
        websocket_connection_manager.config['max_connections_per_user'] = 1
        
        # Connect first WebSocket
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation_1",
            user_id="test_user"
        )
        
        # Try to connect second WebSocket for same user
        mock_websocket2 = AsyncMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.close = AsyncMock()
        mock_websocket2.client_state = WebSocketState.CONNECTED
        mock_websocket2.client = Mock()
        mock_websocket2.client.host = "127.0.0.1"
        mock_websocket2.headers = {"user-agent": "test-client"}
        
        with pytest.raises(Exception, match="Maximum connections per user exceeded"):
            await websocket_connection_manager.connect(
                websocket=mock_websocket2,
                correlation_id="test_correlation_2",
                user_id="test_user"
            )
        
        mock_websocket2.close.assert_called_once_with(code=1008, reason="Connection limit exceeded")
    
    @pytest.mark.asyncio
    async def test_disconnect_success(self, websocket_connection_manager, mock_websocket):
        """Test successful WebSocket disconnection."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation",
            user_id="test_user"
        )
        
        # Subscribe to something
        connection = websocket_connection_manager.connection_manager.get_connection_by_websocket(mock_websocket)
        websocket_connection_manager.subscription_manager.subscribe(
            connection_id=connection.id,
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        # Disconnect
        await websocket_connection_manager.disconnect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        assert len(websocket_connection_manager.connection_manager.connections) == 0
        assert websocket_connection_manager.metrics['active_connections'] == 0
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self, websocket_connection_manager, mock_websocket):
        """Test disconnecting non-existent connection."""
        # Should not raise exception
        await websocket_connection_manager.disconnect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        assert len(websocket_connection_manager.connection_manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, websocket_connection_manager, mock_websocket):
        """Test handling ping message."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Handle ping message
        ping_message = {
            "type": MessageType.PING.value,
            "id": "ping_123"
        }
        
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=ping_message,
            correlation_id="test_correlation"
        )
        
        # Should send pong response
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.PONG.value
        assert call_args["data"]["ping_id"] == "ping_123"
    
    @pytest.mark.asyncio
    async def test_handle_subscription_message(self, websocket_connection_manager, mock_websocket):
        """Test handling subscription message."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Handle subscription message
        subscription_message = {
            "type": MessageType.SUBSCRIPTION.value,
            "subscription_type": SubscriptionType.ALL_EXPLANATIONS.value,
            "filters": {"symbol": "AAPL"}
        }
        
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=subscription_message,
            correlation_id="test_correlation"
        )
        
        # Should send subscription confirmation
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.SUBSCRIPTION.value
        assert call_args["data"]["status"] == "success"
        assert call_args["data"]["subscription_type"] == SubscriptionType.ALL_EXPLANATIONS.value
        
        # Check subscription was created
        assert websocket_connection_manager.metrics['subscription_count'] == 1
    
    @pytest.mark.asyncio
    async def test_handle_invalid_subscription_message(self, websocket_connection_manager, mock_websocket):
        """Test handling invalid subscription message."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Handle invalid subscription message
        invalid_subscription_message = {
            "type": MessageType.SUBSCRIPTION.value,
            "subscription_type": "invalid_subscription_type",
            "filters": {"symbol": "AAPL"}
        }
        
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=invalid_subscription_message,
            correlation_id="test_correlation"
        )
        
        # Should send error response
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.ERROR.value
        assert "Invalid subscription type" in call_args["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_handle_unsubscription_message(self, websocket_connection_manager, mock_websocket):
        """Test handling unsubscription message."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Subscribe first
        connection = websocket_connection_manager.connection_manager.get_connection_by_websocket(mock_websocket)
        websocket_connection_manager.subscription_manager.subscribe(
            connection_id=connection.id,
            subscription_type=SubscriptionType.ALL_EXPLANATIONS
        )
        
        # Handle unsubscription message
        unsubscription_message = {
            "type": MessageType.UNSUBSCRIPTION.value,
            "subscription_type": SubscriptionType.ALL_EXPLANATIONS.value
        }
        
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=unsubscription_message,
            correlation_id="test_correlation"
        )
        
        # Should send unsubscription confirmation
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.UNSUBSCRIPTION.value
        assert call_args["data"]["status"] == "success"
        assert call_args["data"]["removed_count"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, websocket_connection_manager, mock_websocket):
        """Test handling unknown message type."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Handle unknown message type
        unknown_message = {
            "type": "unknown_message_type",
            "data": {"test": "data"}
        }
        
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=unknown_message,
            correlation_id="test_correlation"
        )
        
        # Should send error response
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.ERROR.value
        assert "Unknown message type" in call_args["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, websocket_connection_manager, mock_websocket):
        """Test successful message sending."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Send message
        message_data = {
            "type": MessageType.EXPLANATION.value,
            "explanation": "test explanation"
        }
        
        success = await websocket_connection_manager.send_message(
            websocket=mock_websocket,
            message_data=message_data
        )
        
        assert success is True
        assert websocket_connection_manager.metrics['messages_sent'] == 1
        mock_websocket.send_json.assert_called()
    
    @pytest.mark.asyncio
    async def test_send_message_nonexistent_connection(self, websocket_connection_manager, mock_websocket):
        """Test sending message to non-existent connection."""
        message_data = {
            "type": MessageType.EXPLANATION.value,
            "explanation": "test explanation"
        }
        
        success = await websocket_connection_manager.send_message(
            websocket=mock_websocket,
            message_data=message_data
        )
        
        assert success is False
        assert websocket_connection_manager.metrics['messages_sent'] == 0
    
    @pytest.mark.asyncio
    async def test_send_ping(self, websocket_connection_manager, mock_websocket):
        """Test sending ping message."""
        # Connect first
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Send ping
        success = await websocket_connection_manager.send_ping(mock_websocket)
        
        assert success is True
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.PING.value
    
    @pytest.mark.asyncio
    async def test_broadcast_explanation(self, websocket_connection_manager):
        """Test broadcasting explanation."""
        # Connect multiple WebSockets and subscribe
        mock_websockets = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = "127.0.0.1"
            mock_ws.headers = {"user-agent": "test-client"}
            mock_websockets.append(mock_ws)
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            
            # Subscribe to all explanations
            websocket_connection_manager.subscription_manager.subscribe(
                connection_id=connection_id,
                subscription_type=SubscriptionType.ALL_EXPLANATIONS
            )
        
        # Broadcast explanation
        explanation_data = {
            "explanation": "test explanation",
            "confidence": 0.95
        }
        
        success_count = await websocket_connection_manager.broadcast_explanation(
            explanation_data=explanation_data,
            symbol="AAPL",
            agent="MLMI"
        )
        
        assert success_count == 3
        for mock_ws in mock_websockets:
            mock_ws.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_explanation_with_filters(self, websocket_connection_manager):
        """Test broadcasting explanation with symbol filters."""
        # Connect multiple WebSockets with different symbol subscriptions
        mock_websockets = []
        symbols = ["AAPL", "MSFT", "AAPL"]
        
        for i, symbol in enumerate(symbols):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = "127.0.0.1"
            mock_ws.headers = {"user-agent": "test-client"}
            mock_websockets.append(mock_ws)
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            
            # Subscribe to symbol explanations
            websocket_connection_manager.subscription_manager.subscribe(
                connection_id=connection_id,
                subscription_type=SubscriptionType.SYMBOL_EXPLANATIONS,
                filters={"symbol": symbol}
            )
        
        # Broadcast explanation for AAPL
        explanation_data = {
            "explanation": "test explanation",
            "confidence": 0.95
        }
        
        success_count = await websocket_connection_manager.broadcast_explanation(
            explanation_data=explanation_data,
            symbol="AAPL",
            agent="MLMI"
        )
        
        assert success_count == 2  # Two AAPL subscribers
        mock_websockets[0].send_json.assert_called_once()  # AAPL subscriber
        mock_websockets[1].send_json.assert_not_called()   # MSFT subscriber
        mock_websockets[2].send_json.assert_called_once()  # AAPL subscriber
    
    @pytest.mark.asyncio
    async def test_broadcast_decision_notification(self, websocket_connection_manager):
        """Test broadcasting decision notification."""
        # Connect WebSocket and subscribe
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = WebSocketState.CONNECTED
        mock_ws.client = Mock()
        mock_ws.client.host = "127.0.0.1"
        mock_ws.headers = {"user-agent": "test-client"}
        
        connection_id = await websocket_connection_manager.connect(
            websocket=mock_ws,
            correlation_id="test_correlation"
        )
        
        # Subscribe to decision notifications
        websocket_connection_manager.subscription_manager.subscribe(
            connection_id=connection_id,
            subscription_type=SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        # Broadcast decision notification
        decision_data = {
            "decision": "BUY",
            "confidence": 0.85,
            "symbol": "AAPL"
        }
        
        success_count = await websocket_connection_manager.broadcast_decision_notification(
            decision_data=decision_data
        )
        
        assert success_count == 1
        mock_ws.send_json.assert_called_once()
        
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.DECISION_NOTIFICATION.value
        assert call_args["data"] == decision_data
    
    @pytest.mark.asyncio
    async def test_broadcast_system_status(self, websocket_connection_manager):
        """Test broadcasting system status."""
        # Connect WebSocket and subscribe
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = WebSocketState.CONNECTED
        mock_ws.client = Mock()
        mock_ws.client.host = "127.0.0.1"
        mock_ws.headers = {"user-agent": "test-client"}
        
        connection_id = await websocket_connection_manager.connect(
            websocket=mock_ws,
            correlation_id="test_correlation"
        )
        
        # Subscribe to system status
        websocket_connection_manager.subscription_manager.subscribe(
            connection_id=connection_id,
            subscription_type=SubscriptionType.SYSTEM_STATUS
        )
        
        # Broadcast system status
        status_data = {
            "status": "healthy",
            "cpu_usage": 65.5,
            "memory_usage": 42.1
        }
        
        success_count = await websocket_connection_manager.broadcast_system_status(
            status_data=status_data
        )
        
        assert success_count == 1
        mock_ws.send_json.assert_called_once()
        
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.SYSTEM_STATUS.value
        assert call_args["data"] == status_data
    
    def test_get_connection_count(self, websocket_connection_manager, mock_websocket):
        """Test getting connection count."""
        # Initially zero
        assert websocket_connection_manager.get_connection_count() == 0
        
        # Connect WebSocket
        asyncio.run(websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        ))
        
        assert websocket_connection_manager.get_connection_count() == 1
    
    def test_get_metrics(self, websocket_connection_manager, mock_websocket):
        """Test getting metrics."""
        # Get initial metrics
        metrics = websocket_connection_manager.get_metrics()
        
        assert metrics['total_connections'] == 0
        assert metrics['active_connections'] == 0
        assert metrics['messages_sent'] == 0
        assert metrics['messages_received'] == 0
        assert metrics['subscription_count'] == 0
        assert metrics['total_subscriptions'] == 0
        
        # Connect and test metrics update
        asyncio.run(websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        ))
        
        metrics = websocket_connection_manager.get_metrics()
        assert metrics['total_connections'] == 1
        assert metrics['active_connections'] == 1
    
    @pytest.mark.asyncio
    async def test_shutdown(self, websocket_connection_manager, mock_websocket):
        """Test shutdown process."""
        # Connect WebSocket
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Initialize with mock event bus
        with patch('src.xai.api.websocket_handlers.EventBus') as mock_event_bus_class:
            mock_event_bus = Mock()
            mock_event_bus_class.return_value = mock_event_bus
            
            await websocket_connection_manager.initialize()
            
            # Shutdown
            await websocket_connection_manager.shutdown()
            
            # Check cleanup
            assert len(websocket_connection_manager.connection_manager.connections) == 0
            assert websocket_connection_manager._cleanup_task.cancelled()
            assert websocket_connection_manager._heartbeat_task.cancelled()
            
            # Check event unsubscriptions
            assert mock_event_bus.unsubscribe.call_count == 3

class TestStressTesting:
    """Test stress scenarios for WebSocket handlers."""
    
    @pytest.mark.asyncio
    async def test_many_concurrent_connections(self, websocket_connection_manager):
        """Test handling many concurrent connections."""
        connection_count = 100
        
        # Create many mock WebSockets
        mock_websockets = []
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            mock_websockets.append(mock_ws)
        
        # Connect all WebSockets concurrently
        start_time = time.time()
        
        connection_tasks = []
        for i, mock_ws in enumerate(mock_websockets):
            task = websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}",
                user_id=f"user_{i}"
            )
            connection_tasks.append(task)
        
        connection_ids = await asyncio.gather(*connection_tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 5.0, f"Connection took too long: {duration}s"
        
        # All connections should succeed
        assert len(connection_ids) == connection_count
        assert websocket_connection_manager.get_connection_count() == connection_count
        
        # All WebSockets should be accepted
        for mock_ws in mock_websockets:
            mock_ws.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_high_frequency_messaging(self, websocket_connection_manager):
        """Test high-frequency message handling."""
        # Connect WebSocket
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = WebSocketState.CONNECTED
        mock_ws.client = Mock()
        mock_ws.client.host = "127.0.0.1"
        mock_ws.headers = {"user-agent": "test-client"}
        
        await websocket_connection_manager.connect(
            websocket=mock_ws,
            correlation_id="test_correlation"
        )
        
        # Send many messages rapidly
        message_count = 1000
        messages = []
        
        for i in range(message_count):
            message = {
                "type": MessageType.PING.value,
                "id": f"ping_{i}"
            }
            messages.append(message)
        
        # Send all messages concurrently
        start_time = time.time()
        
        message_tasks = []
        for i, message in enumerate(messages):
            task = websocket_connection_manager.handle_message(
                websocket=mock_ws,
                message_data=message,
                correlation_id=f"test_correlation_{i}"
            )
            message_tasks.append(task)
        
        await asyncio.gather(*message_tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle messages quickly
        assert duration < 10.0, f"Message handling took too long: {duration}s"
        
        # All messages should be processed
        assert mock_ws.send_json.call_count == message_count
        assert websocket_connection_manager.metrics['messages_received'] == message_count
    
    @pytest.mark.asyncio
    async def test_mass_subscription_operations(self, websocket_connection_manager):
        """Test mass subscription operations."""
        # Connect multiple WebSockets
        connection_count = 50
        subscription_types = list(SubscriptionType)
        
        mock_websockets = []
        connection_ids = []
        
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            mock_websockets.append(mock_ws)
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            connection_ids.append(connection_id)
        
        # Subscribe all connections to all subscription types
        subscription_tasks = []
        for connection_id in connection_ids:
            for subscription_type in subscription_types:
                subscription_tasks.append(
                    asyncio.create_task(
                        asyncio.coroutine(lambda: websocket_connection_manager.subscription_manager.subscribe(
                            connection_id=connection_id,
                            subscription_type=subscription_type
                        ))()
                    )
                )
        
        start_time = time.time()
        await asyncio.gather(*subscription_tasks)
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 5.0, f"Subscription operations took too long: {duration}s"
        
        # Check subscription counts
        total_subscriptions = connection_count * len(subscription_types)
        metrics = websocket_connection_manager.get_metrics()
        assert metrics['total_subscriptions'] == total_subscriptions
    
    @pytest.mark.asyncio
    async def test_broadcast_performance_under_load(self, websocket_connection_manager):
        """Test broadcast performance under load."""
        # Connect many WebSockets and subscribe
        connection_count = 200
        mock_websockets = []
        
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            mock_websockets.append(mock_ws)
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            
            # Subscribe to all explanations
            websocket_connection_manager.subscription_manager.subscribe(
                connection_id=connection_id,
                subscription_type=SubscriptionType.ALL_EXPLANATIONS
            )
        
        # Broadcast explanation to all connections
        explanation_data = {
            "explanation": "test explanation",
            "confidence": 0.95
        }
        
        start_time = time.time()
        
        success_count = await websocket_connection_manager.broadcast_explanation(
            explanation_data=explanation_data,
            symbol="AAPL",
            agent="MLMI"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 2.0, f"Broadcast took too long: {duration}s"
        
        # All connections should receive the message
        assert success_count == connection_count
        
        for mock_ws in mock_websockets:
            mock_ws.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_performance(self, websocket_connection_manager):
        """Test connection cleanup performance."""
        # Connect many WebSockets
        connection_count = 100
        mock_websockets = []
        
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.close = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            mock_websockets.append(mock_ws)
            
            await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
        
        # Disconnect all connections
        disconnect_tasks = []
        for mock_ws in mock_websockets:
            task = websocket_connection_manager.disconnect(
                websocket=mock_ws,
                correlation_id="cleanup_test"
            )
            disconnect_tasks.append(task)
        
        start_time = time.time()
        await asyncio.gather(*disconnect_tasks)
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 3.0, f"Cleanup took too long: {duration}s"
        
        # All connections should be cleaned up
        assert websocket_connection_manager.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, websocket_connection_manager):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Connect many WebSockets with subscriptions
        connection_count = 500
        
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            
            # Add multiple subscriptions
            for sub_type in [SubscriptionType.ALL_EXPLANATIONS, SubscriptionType.DECISION_NOTIFICATIONS]:
                websocket_connection_manager.subscription_manager.subscribe(
                    connection_id=connection_id,
                    subscription_type=sub_type
                )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB for 500 connections)
        assert memory_increase < 200 * 1024 * 1024, f"Memory usage too high: {memory_increase / (1024*1024):.1f} MB"
    
    @pytest.mark.asyncio
    async def test_error_recovery_under_load(self, websocket_connection_manager):
        """Test error recovery under load."""
        # Connect WebSockets, some of which will fail
        connection_count = 50
        error_count = 10
        
        mock_websockets = []
        for i in range(connection_count):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i % 254 + 1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            
            # Some WebSockets will fail to send
            if i < error_count:
                mock_ws.send_json = AsyncMock(side_effect=Exception("Send failed"))
            else:
                mock_ws.send_json = AsyncMock()
            
            mock_websockets.append(mock_ws)
            
            connection_id = await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}"
            )
            
            # Subscribe to explanations
            websocket_connection_manager.subscription_manager.subscribe(
                connection_id=connection_id,
                subscription_type=SubscriptionType.ALL_EXPLANATIONS
            )
        
        # Broadcast message (some will fail)
        explanation_data = {
            "explanation": "test explanation",
            "confidence": 0.95
        }
        
        success_count = await websocket_connection_manager.broadcast_explanation(
            explanation_data=explanation_data,
            symbol="AAPL",
            agent="MLMI"
        )
        
        # Should handle errors gracefully
        expected_success_count = connection_count - error_count
        assert success_count == expected_success_count
        
        # Failed connections should be cleaned up
        assert websocket_connection_manager.get_connection_count() == expected_success_count

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect_during_send(self, websocket_connection_manager, mock_websocket):
        """Test WebSocket disconnect during message send."""
        # Connect WebSocket
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Simulate WebSocket disconnect during send
        mock_websocket.client_state = WebSocketState.DISCONNECTED
        
        # Try to send message
        message_data = {
            "type": MessageType.EXPLANATION.value,
            "explanation": "test explanation"
        }
        
        success = await websocket_connection_manager.send_message(
            websocket=mock_websocket,
            message_data=message_data
        )
        
        assert success is False
        # Connection should be cleaned up
        assert websocket_connection_manager.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_message_processing_exception(self, websocket_connection_manager, mock_websocket):
        """Test exception during message processing."""
        # Connect WebSocket
        await websocket_connection_manager.connect(
            websocket=mock_websocket,
            correlation_id="test_correlation"
        )
        
        # Send malformed message that will cause processing error
        malformed_message = {
            "type": MessageType.SUBSCRIPTION.value,
            "subscription_type": None,  # This will cause an error
            "filters": {"symbol": "AAPL"}
        }
        
        # Should handle error gracefully
        await websocket_connection_manager.handle_message(
            websocket=mock_websocket,
            message_data=malformed_message,
            correlation_id="test_correlation"
        )
        
        # Should send error response
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == MessageType.ERROR.value
        assert "Message processing error" in call_args["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_connection_limit_handling(self, websocket_connection_manager):
        """Test connection limit handling."""
        # Set very low connection limit
        websocket_connection_manager.config['max_connections_per_user'] = 2
        
        # Connect up to the limit
        mock_websockets = []
        for i in range(2):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.client_state = WebSocketState.CONNECTED
            mock_ws.client = Mock()
            mock_ws.client.host = "127.0.0.1"
            mock_ws.headers = {"user-agent": "test-client"}
            mock_websockets.append(mock_ws)
            
            await websocket_connection_manager.connect(
                websocket=mock_ws,
                correlation_id=f"test_correlation_{i}",
                user_id="test_user"
            )
        
        # Try to connect one more (should fail)
        mock_ws_extra = AsyncMock(spec=WebSocket)
        mock_ws_extra.accept = AsyncMock()
        mock_ws_extra.close = AsyncMock()
        mock_ws_extra.client_state = WebSocketState.CONNECTED
        mock_ws_extra.client = Mock()
        mock_ws_extra.client.host = "127.0.0.1"
        mock_ws_extra.headers = {"user-agent": "test-client"}
        
        with pytest.raises(Exception, match="Maximum connections per user exceeded"):
            await websocket_connection_manager.connect(
                websocket=mock_ws_extra,
                correlation_id="test_correlation_extra",
                user_id="test_user"
            )
        
        # Should close the extra WebSocket
        mock_ws_extra.close.assert_called_once()
        
        # Only 2 connections should remain
        assert websocket_connection_manager.get_connection_count() == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])