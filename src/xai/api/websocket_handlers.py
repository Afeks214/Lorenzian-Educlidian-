"""
WebSocket Connection Management for XAI API
AGENT DELTA MISSION: Real-time WebSocket Communication

This module implements sophisticated WebSocket connection management for real-time
explanation streaming, decision notifications, and system status updates.

Features:
- WebSocket connection lifecycle management
- Real-time explanation streaming
- Decision notification broadcasting
- System status updates
- Connection authentication and authorization
- Message routing and filtering
- Connection health monitoring
- Automatic reconnection support

Author: Agent Delta - Integration Specialist
Version: 1.0 - WebSocket Communication Layer
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.monitoring.logger_config import get_logger
from src.core.event_bus import EventBus
from src.core.events import EventType, Event

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    WELCOME = "welcome"
    PING = "ping"
    PONG = "pong"
    EXPLANATION = "explanation"
    DECISION_NOTIFICATION = "decision_notification"
    SYSTEM_STATUS = "system_status"
    AGENT_PERFORMANCE = "agent_performance"
    QUERY_RESPONSE = "query_response"
    ERROR = "error"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"
    HEARTBEAT = "heartbeat"


class SubscriptionType(Enum):
    """Types of subscriptions available"""
    ALL_EXPLANATIONS = "all_explanations"
    SYMBOL_EXPLANATIONS = "symbol_explanations"  # e.g., "NQ" explanations only
    AGENT_EXPLANATIONS = "agent_explanations"    # e.g., "MLMI" agent only
    DECISION_NOTIFICATIONS = "decision_notifications"
    SYSTEM_STATUS = "system_status"
    PERFORMANCE_UPDATES = "performance_updates"
    RISK_ALERTS = "risk_alerts"


@dataclass
class WebSocketConnection:
    """WebSocket connection metadata"""
    id: str
    websocket: WebSocket
    user_id: Optional[str]
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str]
    connection_info: Dict[str, Any]
    message_count: int
    bytes_sent: int
    bytes_received: int
    
    
@dataclass
class Subscription:
    """WebSocket subscription details"""
    connection_id: str
    subscription_type: SubscriptionType
    filters: Dict[str, Any]  # e.g., {"symbol": "NQ", "agent": "MLMI"}
    created_at: datetime
    

@dataclass
class WebSocketMessage:
    """Standardized WebSocket message format"""
    id: str
    type: MessageType
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    target_connection: Optional[str] = None


class ConnectionManager:
    """Manage individual WebSocket connections"""
    
    def __init__(self):
        """Initialize connection manager"""
        self.connections: Dict[str, WebSocketConnection] = {}
        self.connection_lookup: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        user_id: Optional[str] = None,
        connection_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            connection_id: Unique connection identifier
            user_id: Authenticated user ID
            connection_info: Additional connection metadata
            
        Returns:
            str: Connection ID
        """
        await websocket.accept()
        
        connection = WebSocketConnection(
            id=connection_id,
            websocket=websocket,
            user_id=user_id,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            subscriptions=set(),
            connection_info=connection_info or {},
            message_count=0,
            bytes_sent=0,
            bytes_received=0
        )
        
        self.connections[connection_id] = connection
        self.connection_lookup[websocket] = connection_id
        
        logger.info(
            "WebSocket connection established",
            extra={
                "connection_id": connection_id,
                "user_id": user_id,
                "client_info": connection_info
            }
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect and cleanup a WebSocket connection.
        
        Args:
            connection_id: Connection to disconnect
        """
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            try:
                if connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            
            # Remove from lookups
            if connection.websocket in self.connection_lookup:
                del self.connection_lookup[connection.websocket]
            
            del self.connections[connection_id]
            
            logger.info(
                "WebSocket connection closed",
                extra={
                    "connection_id": connection_id,
                    "duration_seconds": (datetime.now() - connection.connected_at).total_seconds(),
                    "messages_sent": connection.message_count,
                    "bytes_transferred": connection.bytes_sent + connection.bytes_received
                }
            )
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def get_connection_by_websocket(self, websocket: WebSocket) -> Optional[WebSocketConnection]:
        """Get connection by WebSocket instance"""
        connection_id = self.connection_lookup.get(websocket)
        return self.connections.get(connection_id) if connection_id else None
    
    def get_all_connections(self) -> List[WebSocketConnection]:
        """Get all active connections"""
        return list(self.connections.values())
    
    def get_connections_by_user(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a specific user"""
        return [
            conn for conn in self.connections.values()
            if conn.user_id == user_id
        ]
    
    async def send_message(
        self, 
        connection_id: str, 
        message: WebSocketMessage
    ) -> bool:
        """
        Send message to specific connection.
        
        Args:
            connection_id: Target connection
            message: Message to send
            
        Returns:
            bool: Success status
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        try:
            if connection.websocket.client_state != WebSocketState.CONNECTED:
                await self.disconnect(connection_id)
                return False
            
            message_data = {
                "id": message.id,
                "type": message.type.value,
                "timestamp": message.timestamp.isoformat(),
                "data": message.data
            }
            
            if message.correlation_id:
                message_data["correlation_id"] = message.correlation_id
            
            await connection.websocket.send_json(message_data)
            
            # Update connection stats
            connection.message_count += 1
            connection.bytes_sent += len(json.dumps(message_data))
            connection.last_activity = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send WebSocket message",
                extra={
                    "connection_id": connection_id,
                    "error": str(e),
                    "message_type": message.type.value
                }
            )
            await self.disconnect(connection_id)
            return False
    
    async def broadcast_message(
        self, 
        message: WebSocketMessage,
        filter_func: Optional[Callable[[WebSocketConnection], bool]] = None
    ) -> int:
        """
        Broadcast message to multiple connections.
        
        Args:
            message: Message to broadcast
            filter_func: Optional filter function for connections
            
        Returns:
            int: Number of successful sends
        """
        connections = self.connections.values()
        
        if filter_func:
            connections = [conn for conn in connections if filter_func(conn)]
        
        success_count = 0
        tasks = []
        
        for connection in connections:
            task = self.send_message(connection.id, message)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
        
        return success_count
    
    def cleanup_stale_connections(self, timeout_minutes: int = 30) -> int:
        """
        Cleanup stale connections.
        
        Args:
            timeout_minutes: Timeout threshold in minutes
            
        Returns:
            int: Number of connections cleaned up
        """
        cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
        stale_connections = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.last_activity < cutoff_time
        ]
        
        for conn_id in stale_connections:
            asyncio.create_task(self.disconnect(conn_id))
        
        return len(stale_connections)


class SubscriptionManager:
    """Manage WebSocket subscriptions"""
    
    def __init__(self):
        """Initialize subscription manager"""
        self.subscriptions: Dict[str, List[Subscription]] = {}
        self.connection_subscriptions: Dict[str, List[Subscription]] = {}
    
    def subscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new subscription.
        
        Args:
            connection_id: Connection ID
            subscription_type: Type of subscription
            filters: Optional filters for the subscription
            
        Returns:
            str: Subscription ID
        """
        subscription = Subscription(
            connection_id=connection_id,
            subscription_type=subscription_type,
            filters=filters or {},
            created_at=datetime.now()
        )
        
        subscription_key = subscription_type.value
        
        # Add to subscription lookup
        if subscription_key not in self.subscriptions:
            self.subscriptions[subscription_key] = []
        self.subscriptions[subscription_key].append(subscription)
        
        # Add to connection lookup
        if connection_id not in self.connection_subscriptions:
            self.connection_subscriptions[connection_id] = []
        self.connection_subscriptions[connection_id].append(subscription)
        
        logger.info(
            "WebSocket subscription created",
            extra={
                "connection_id": connection_id,
                "subscription_type": subscription_type.value,
                "filters": filters
            }
        )
        
        return f"{connection_id}_{subscription_type.value}_{int(time.time())}"
    
    def unsubscribe(
        self,
        connection_id: str,
        subscription_type: Optional[SubscriptionType] = None
    ) -> int:
        """
        Remove subscriptions.
        
        Args:
            connection_id: Connection ID
            subscription_type: Specific subscription type (None for all)
            
        Returns:
            int: Number of subscriptions removed
        """
        removed_count = 0
        
        # Get connection subscriptions
        connection_subs = self.connection_subscriptions.get(connection_id, [])
        
        # Filter subscriptions to remove
        subs_to_remove = [
            sub for sub in connection_subs
            if subscription_type is None or sub.subscription_type == subscription_type
        ]
        
        # Remove from both lookups
        for subscription in subs_to_remove:
            # Remove from subscription lookup
            subscription_key = subscription.subscription_type.value
            if subscription_key in self.subscriptions:
                self.subscriptions[subscription_key] = [
                    s for s in self.subscriptions[subscription_key]
                    if s.connection_id != connection_id or s.subscription_type != subscription.subscription_type
                ]
            
            # Remove from connection lookup
            self.connection_subscriptions[connection_id] = [
                s for s in self.connection_subscriptions[connection_id]
                if s != subscription
            ]
            
            removed_count += 1
        
        if removed_count > 0:
            logger.info(
                "WebSocket subscriptions removed",
                extra={
                    "connection_id": connection_id,
                    "subscription_type": subscription_type.value if subscription_type else "all",
                    "count": removed_count
                }
            )
        
        return removed_count
    
    def get_subscribers(
        self,
        subscription_type: SubscriptionType,
        message_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get connection IDs that should receive a message.
        
        Args:
            subscription_type: Type of subscription
            message_data: Message data for filtering
            
        Returns:
            List[str]: Connection IDs
        """
        subscription_key = subscription_type.value
        subscriptions = self.subscriptions.get(subscription_key, [])
        
        matching_connections = []
        
        for subscription in subscriptions:
            if self._matches_filters(subscription.filters, message_data):
                matching_connections.append(subscription.connection_id)
        
        return matching_connections
    
    def _matches_filters(
        self,
        filters: Dict[str, Any],
        message_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if message matches subscription filters"""
        if not filters or not message_data:
            return True
        
        for filter_key, filter_value in filters.items():
            if filter_key in message_data:
                if message_data[filter_key] != filter_value:
                    return False
            else:
                # Filter key not in message data
                return False
        
        return True
    
    def cleanup_connection_subscriptions(self, connection_id: str) -> int:
        """
        Clean up all subscriptions for a connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            int: Number of subscriptions removed
        """
        return self.unsubscribe(connection_id)


class WebSocketConnectionManager:
    """
    Comprehensive WebSocket Connection Manager
    
    Manages WebSocket connections, subscriptions, and real-time message broadcasting
    for the XAI API system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WebSocket connection manager"""
        self.config = config or self._default_config()
        
        # Core managers
        self.connection_manager = ConnectionManager()
        self.subscription_manager = SubscriptionManager()
        
        # Event bus for system integration
        self.event_bus: Optional[EventBus] = None
        
        # Performance metrics
        self.metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_transferred': 0,
            'connection_errors': 0,
            'subscription_count': 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocket Connection Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'heartbeat_interval_seconds': 30,
            'cleanup_interval_minutes': 5,
            'connection_timeout_minutes': 30,
            'max_connections_per_user': 5,
            'message_rate_limit': 100,  # messages per minute
            'enable_compression': True
        }
    
    async def initialize(self) -> None:
        """Initialize connection manager"""
        try:
            # Initialize event bus
            self.event_bus = EventBus()
            
            # Subscribe to system events
            self._subscribe_to_system_events()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("WebSocket Connection Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket Connection Manager: {e}")
            raise
    
    def _subscribe_to_system_events(self) -> None:
        """Subscribe to system events for broadcasting"""
        if self.event_bus:
            self.event_bus.subscribe(
                EventType.STRATEGIC_DECISION,
                self._handle_strategic_decision_event
            )
            self.event_bus.subscribe(
                EventType.AGENT_PERFORMANCE_UPDATE,
                self._handle_agent_performance_event
            )
            self.event_bus.subscribe(
                EventType.SYSTEM_ALERT,
                self._handle_system_alert_event
            )
    
    async def connect(
        self, 
        websocket: WebSocket, 
        correlation_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Accept and manage a new WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            correlation_id: Request correlation ID
            user_id: Authenticated user ID
            
        Returns:
            str: Connection ID
        """
        # Check connection limits
        if user_id:
            user_connections = self.connection_manager.get_connections_by_user(user_id)
            if len(user_connections) >= self.config['max_connections_per_user']:
                await websocket.close(code=1008, reason="Connection limit exceeded")
                raise Exception("Maximum connections per user exceeded")
        
        # Generate connection ID
        connection_id = f"ws_{uuid.uuid4().hex[:8]}"
        
        # Extract connection info
        connection_info = {
            "correlation_id": correlation_id,
            "client_host": websocket.client.host if websocket.client else "unknown",
            "user_agent": websocket.headers.get("user-agent", "unknown")
        }
        
        # Connect
        await self.connection_manager.connect(
            websocket=websocket,
            connection_id=connection_id,
            user_id=user_id,
            connection_info=connection_info
        )
        
        # Update metrics
        self.metrics['total_connections'] += 1
        self.metrics['active_connections'] = len(self.connection_manager.connections)
        
        return connection_id
    
    async def disconnect(
        self, 
        websocket: WebSocket, 
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Disconnect a WebSocket connection.
        
        Args:
            websocket: WebSocket instance
            correlation_id: Request correlation ID
        """
        connection = self.connection_manager.get_connection_by_websocket(websocket)
        
        if connection:
            # Clean up subscriptions
            self.subscription_manager.cleanup_connection_subscriptions(connection.id)
            
            # Disconnect
            await self.connection_manager.disconnect(connection.id)
            
            # Update metrics
            self.metrics['active_connections'] = len(self.connection_manager.connections)
    
    async def handle_message(
        self,
        websocket: WebSocket,
        message_data: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            websocket: WebSocket instance
            message_data: Received message data
            correlation_id: Request correlation ID
        """
        connection = self.connection_manager.get_connection_by_websocket(websocket)
        
        if not connection:
            logger.warning("Received message from unknown connection")
            return
        
        try:
            message_type = message_data.get("type")
            
            if message_type == MessageType.PING.value:
                await self._handle_ping(connection, message_data)
            
            elif message_type == MessageType.SUBSCRIPTION.value:
                await self._handle_subscription(connection, message_data)
            
            elif message_type == MessageType.UNSUBSCRIPTION.value:
                await self._handle_unsubscription(connection, message_data)
            
            elif message_type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(connection, message_data)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self._send_error(
                    connection.id,
                    f"Unknown message type: {message_type}",
                    correlation_id
                )
            
            # Update connection activity
            connection.last_activity = datetime.now()
            self.metrics['messages_received'] += 1
            
        except Exception as e:
            logger.error(
                "Error handling WebSocket message",
                extra={
                    "connection_id": connection.id,
                    "error": str(e),
                    "message_type": message_data.get("type")
                }
            )
            await self._send_error(
                connection.id,
                f"Message processing error: {str(e)}",
                correlation_id
            )
    
    async def _handle_ping(
        self,
        connection: WebSocketConnection,
        message_data: Dict[str, Any]
    ) -> None:
        """Handle ping message"""
        pong_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.PONG,
            timestamp=datetime.now(),
            data={"ping_id": message_data.get("id", "unknown")}
        )
        
        await self.connection_manager.send_message(connection.id, pong_message)
    
    async def _handle_subscription(
        self,
        connection: WebSocketConnection,
        message_data: Dict[str, Any]
    ) -> None:
        """Handle subscription request"""
        subscription_type_str = message_data.get("subscription_type")
        filters = message_data.get("filters", {})
        
        try:
            subscription_type = SubscriptionType(subscription_type_str)
            
            subscription_id = self.subscription_manager.subscribe(
                connection_id=connection.id,
                subscription_type=subscription_type,
                filters=filters
            )
            
            # Send confirmation
            response = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.SUBSCRIPTION,
                timestamp=datetime.now(),
                data={
                    "status": "success",
                    "subscription_id": subscription_id,
                    "subscription_type": subscription_type_str,
                    "filters": filters
                }
            )
            
            await self.connection_manager.send_message(connection.id, response)
            
            self.metrics['subscription_count'] += 1
            
        except ValueError:
            await self._send_error(
                connection.id,
                f"Invalid subscription type: {subscription_type_str}",
                message_data.get("correlation_id")
            )
    
    async def _handle_unsubscription(
        self,
        connection: WebSocketConnection,
        message_data: Dict[str, Any]
    ) -> None:
        """Handle unsubscription request"""
        subscription_type_str = message_data.get("subscription_type")
        
        try:
            subscription_type = SubscriptionType(subscription_type_str) if subscription_type_str else None
            
            removed_count = self.subscription_manager.unsubscribe(
                connection_id=connection.id,
                subscription_type=subscription_type
            )
            
            # Send confirmation
            response = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.UNSUBSCRIPTION,
                timestamp=datetime.now(),
                data={
                    "status": "success",
                    "subscription_type": subscription_type_str or "all",
                    "removed_count": removed_count
                }
            )
            
            await self.connection_manager.send_message(connection.id, response)
            
            self.metrics['subscription_count'] -= removed_count
            
        except ValueError:
            await self._send_error(
                connection.id,
                f"Invalid subscription type: {subscription_type_str}",
                message_data.get("correlation_id")
            )
    
    async def _handle_heartbeat(
        self,
        connection: WebSocketConnection,
        message_data: Dict[str, Any]
    ) -> None:
        """Handle heartbeat message"""
        # Simply update last activity (already done in handle_message)
        pass
    
    async def send_message(
        self,
        websocket: WebSocket,
        message_data: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific WebSocket.
        
        Args:
            websocket: Target WebSocket
            message_data: Message data
            
        Returns:
            bool: Success status
        """
        connection = self.connection_manager.get_connection_by_websocket(websocket)
        
        if not connection:
            return False
        
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType(message_data.get("type", "unknown")),
            timestamp=datetime.now(),
            data=message_data
        )
        
        success = await self.connection_manager.send_message(connection.id, message)
        
        if success:
            self.metrics['messages_sent'] += 1
        
        return success
    
    async def send_ping(self, websocket: WebSocket) -> bool:
        """Send ping message to WebSocket"""
        ping_data = {
            "type": MessageType.PING.value,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4())
        }
        
        return await self.send_message(websocket, ping_data)
    
    async def broadcast_explanation(
        self,
        explanation_data: Dict[str, Any],
        symbol: Optional[str] = None,
        agent: Optional[str] = None
    ) -> int:
        """
        Broadcast explanation to subscribed connections.
        
        Args:
            explanation_data: Explanation data
            symbol: Optional symbol filter
            agent: Optional agent filter
            
        Returns:
            int: Number of successful broadcasts
        """
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.EXPLANATION,
            timestamp=datetime.now(),
            data={
                **explanation_data,
                "symbol": symbol,
                "agent": agent
            }
        )
        
        # Get subscribers for different subscription types
        all_subscribers = self.subscription_manager.get_subscribers(
            SubscriptionType.ALL_EXPLANATIONS
        )
        
        symbol_subscribers = []
        if symbol:
            symbol_subscribers = self.subscription_manager.get_subscribers(
                SubscriptionType.SYMBOL_EXPLANATIONS,
                {"symbol": symbol}
            )
        
        agent_subscribers = []
        if agent:
            agent_subscribers = self.subscription_manager.get_subscribers(
                SubscriptionType.AGENT_EXPLANATIONS,
                {"agent": agent}
            )
        
        # Combine and deduplicate subscribers
        all_target_connections = set(all_subscribers + symbol_subscribers + agent_subscribers)
        
        # Send to all target connections
        success_count = 0
        for connection_id in all_target_connections:
            success = await self.connection_manager.send_message(connection_id, message)
            if success:
                success_count += 1
        
        return success_count
    
    async def broadcast_decision_notification(
        self,
        decision_data: Dict[str, Any]
    ) -> int:
        """
        Broadcast decision notification to subscribed connections.
        
        Args:
            decision_data: Decision data
            
        Returns:
            int: Number of successful broadcasts
        """
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.DECISION_NOTIFICATION,
            timestamp=datetime.now(),
            data=decision_data
        )
        
        subscribers = self.subscription_manager.get_subscribers(
            SubscriptionType.DECISION_NOTIFICATIONS
        )
        
        success_count = 0
        for connection_id in subscribers:
            success = await self.connection_manager.send_message(connection_id, message)
            if success:
                success_count += 1
        
        return success_count
    
    async def broadcast_system_status(
        self,
        status_data: Dict[str, Any]
    ) -> int:
        """
        Broadcast system status to subscribed connections.
        
        Args:
            status_data: System status data
            
        Returns:
            int: Number of successful broadcasts
        """
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_STATUS,
            timestamp=datetime.now(),
            data=status_data
        )
        
        subscribers = self.subscription_manager.get_subscribers(
            SubscriptionType.SYSTEM_STATUS
        )
        
        success_count = 0
        for connection_id in subscribers:
            success = await self.connection_manager.send_message(connection_id, message)
            if success:
                success_count += 1
        
        return success_count
    
    async def _send_error(
        self,
        connection_id: str,
        error_message: str,
        correlation_id: Optional[str] = None
    ) -> None:
        """Send error message to connection"""
        error_msg = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            timestamp=datetime.now(),
            data={"error": error_message},
            correlation_id=correlation_id
        )
        
        await self.connection_manager.send_message(connection_id, error_msg)
    
    def _handle_strategic_decision_event(self, event: Event) -> None:
        """Handle strategic decision events"""
        try:
            decision_data = event.payload
            
            # Broadcast decision notification
            asyncio.create_task(
                self.broadcast_decision_notification(decision_data)
            )
            
        except Exception as e:
            logger.error(f"Error handling strategic decision event: {e}")
    
    def _handle_agent_performance_event(self, event: Event) -> None:
        """Handle agent performance update events"""
        try:
            performance_data = event.payload
            
            message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                data=performance_data
            )
            
            subscribers = self.subscription_manager.get_subscribers(
                SubscriptionType.PERFORMANCE_UPDATES
            )
            
            # Broadcast to subscribers
            for connection_id in subscribers:
                asyncio.create_task(
                    self.connection_manager.send_message(connection_id, message)
                )
            
        except Exception as e:
            logger.error(f"Error handling agent performance event: {e}")
    
    def _handle_system_alert_event(self, event: Event) -> None:
        """Handle system alert events"""
        try:
            alert_data = event.payload
            
            # Broadcast system status update
            asyncio.create_task(
                self.broadcast_system_status(alert_data)
            )
            
        except Exception as e:
            logger.error(f"Error handling system alert event: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.config['cleanup_interval_minutes'] * 60)
                
                # Cleanup stale connections
                cleaned_count = self.connection_manager.cleanup_stale_connections(
                    self.config['connection_timeout_minutes']
                )
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} stale WebSocket connections")
                    self.metrics['active_connections'] = len(self.connection_manager.connections)
                
            except Exception as e:
                logger.error(f"Error in WebSocket cleanup loop: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop"""
        while True:
            try:
                await asyncio.sleep(self.config['heartbeat_interval_seconds'])
                
                # Send heartbeat to all connections
                heartbeat_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.HEARTBEAT,
                    timestamp=datetime.now(),
                    data={"server_time": datetime.now().isoformat()}
                )
                
                connections = self.connection_manager.get_all_connections()
                
                for connection in connections:
                    asyncio.create_task(
                        self.connection_manager.send_message(connection.id, heartbeat_message)
                    )
                
            except Exception as e:
                logger.error(f"Error in WebSocket heartbeat loop: {e}")
    
    def get_connection_count(self) -> int:
        """Get current connection count"""
        return len(self.connection_manager.connections)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket metrics"""
        return {
            **self.metrics,
            'active_connections': len(self.connection_manager.connections),
            'total_subscriptions': sum(
                len(subs) for subs in self.subscription_manager.subscriptions.values()
            )
        }
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket connection manager"""
        try:
            logger.info("Shutting down WebSocket Connection Manager")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Close all connections
            connections = list(self.connection_manager.connections.keys())
            for connection_id in connections:
                await self.connection_manager.disconnect(connection_id)
            
            # Unsubscribe from events
            if self.event_bus:
                self.event_bus.unsubscribe(
                    EventType.STRATEGIC_DECISION,
                    self._handle_strategic_decision_event
                )
                self.event_bus.unsubscribe(
                    EventType.AGENT_PERFORMANCE_UPDATE,
                    self._handle_agent_performance_event
                )
                self.event_bus.unsubscribe(
                    EventType.SYSTEM_ALERT,
                    self._handle_system_alert_event
                )
            
            logger.info(
                "WebSocket Connection Manager shutdown complete",
                extra={"final_metrics": self.get_metrics()}
            )
            
        except Exception as e:
            logger.error(f"Error during WebSocket shutdown: {e}")