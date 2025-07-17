"""
Real-time Status Updates System
===============================

WebSocket-based real-time status updates for system control dashboard including:
- WebSocket connection management
- Real-time system status broadcasting
- Component health monitoring
- Performance metrics streaming
- Alert notifications
- Connection recovery and reconnection
- Message queuing and delivery guarantees
- Subscription management

This module provides reliable real-time communication between the system
and the dashboard for immediate status updates and notifications.
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import time
import uuid
from collections import defaultdict, deque

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from pydantic import BaseModel, Field
import jwt

from src.api.authentication import UserInfo, verify_token
from src.api.system_status_endpoints import system_monitor
from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# WebSocket message types
class MessageType(str, Enum):
    """WebSocket message types"""
    SYSTEM_STATUS = "system_status"
    COMPONENT_HEALTH = "component_health"
    PERFORMANCE_METRICS = "performance_metrics"
    ALERT_NOTIFICATION = "alert_notification"
    AUDIT_LOG = "audit_log"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION_ACK = "subscription_ack"
    ERROR = "error"
    RECONNECT = "reconnect"

class SubscriptionType(str, Enum):
    """Subscription types"""
    SYSTEM_STATUS = "system_status"
    COMPONENT_HEALTH = "component_health"
    PERFORMANCE_METRICS = "performance_metrics"
    ALERTS = "alerts"
    AUDIT_LOGS = "audit_logs"
    ALL = "all"

class ConnectionStatus(str, Enum):
    """Connection status"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "correlation_id": self.correlation_id
        }

@dataclass
class ClientConnection:
    """Client connection information"""
    connection_id: str
    websocket: WebSocket
    user_info: UserInfo
    subscriptions: Set[SubscriptionType]
    last_heartbeat: datetime
    connected_at: datetime
    message_queue: deque
    status: ConnectionStatus
    
class RealTimeUpdateManager:
    """
    Real-time update manager for WebSocket connections
    """
    
    def __init__(self):
        self.connections: Dict[str, ClientConnection] = {}
        self.subscription_groups: Dict[SubscriptionType, Set[str]] = defaultdict(set)
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.heartbeat_interval = 30  # seconds
        self.message_queue_size = 1000
        self.connection_timeout = 60  # seconds
        self.running = False
        
        # Initialize message handlers
        self._setup_message_handlers()
        
    async def init_redis(self):
        """Initialize Redis connection for pub/sub"""
        if not self.redis_client:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.message_handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.SUBSCRIPTION_ACK: self._handle_subscription_ack,
            MessageType.ERROR: self._handle_error
        }
    
    async def start(self):
        """Start the real-time update manager"""
        self.running = True
        await self.init_redis()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._system_status_broadcaster())
        asyncio.create_task(self._performance_metrics_broadcaster())
        asyncio.create_task(self._alert_broadcaster())
        asyncio.create_task(self._redis_subscriber())
        
        logger.info("Real-time update manager started")
    
    async def stop(self):
        """Stop the real-time update manager"""
        self.running = False
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self._disconnect_client(connection.connection_id)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Real-time update manager stopped")
    
    async def connect_client(self, websocket: WebSocket, user_info: UserInfo) -> str:
        """Connect a new client"""
        connection_id = str(uuid.uuid4())
        
        connection = ClientConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_info=user_info,
            subscriptions=set(),
            last_heartbeat=datetime.utcnow(),
            connected_at=datetime.utcnow(),
            message_queue=deque(maxlen=self.message_queue_size),
            status=ConnectionStatus.CONNECTED
        )
        
        self.connections[connection_id] = connection
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            type=MessageType.SUBSCRIPTION_ACK,
            data={
                "connection_id": connection_id,
                "user_id": user_info.user_id,
                "username": user_info.username,
                "connected_at": connection.connected_at.isoformat(),
                "available_subscriptions": [sub.value for sub in SubscriptionType]
            },
            timestamp=datetime.utcnow(),
            message_id=str(uuid.uuid4())
        )
        
        await self._send_message(connection_id, welcome_message)
        
        logger.info(f"Client connected: {connection_id} (user: {user_info.username})")
        
        return connection_id
    
    async def disconnect_client(self, connection_id: str):
        """Disconnect a client"""
        await self._disconnect_client(connection_id)
    
    async def _disconnect_client(self, connection_id: str):
        """Internal disconnect client"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from subscription groups
        for subscription in connection.subscriptions:
            self.subscription_groups[subscription].discard(connection_id)
        
        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"Client disconnected: {connection_id}")
    
    async def subscribe_client(self, connection_id: str, subscription_type: SubscriptionType):
        """Subscribe client to updates"""
        if connection_id not in self.connections:
            logger.warning(f"Unknown connection: {connection_id}")
            return
        
        connection = self.connections[connection_id]
        connection.subscriptions.add(subscription_type)
        self.subscription_groups[subscription_type].add(connection_id)
        
        # Send subscription acknowledgment
        ack_message = WebSocketMessage(
            type=MessageType.SUBSCRIPTION_ACK,
            data={
                "subscription_type": subscription_type.value,
                "subscribed": True,
                "active_subscriptions": list(connection.subscriptions)
            },
            timestamp=datetime.utcnow(),
            message_id=str(uuid.uuid4())
        )
        
        await self._send_message(connection_id, ack_message)
        
        logger.info(f"Client {connection_id} subscribed to {subscription_type.value}")
    
    async def unsubscribe_client(self, connection_id: str, subscription_type: SubscriptionType):
        """Unsubscribe client from updates"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.subscriptions.discard(subscription_type)
        self.subscription_groups[subscription_type].discard(connection_id)
        
        # Send unsubscription acknowledgment
        ack_message = WebSocketMessage(
            type=MessageType.SUBSCRIPTION_ACK,
            data={
                "subscription_type": subscription_type.value,
                "subscribed": False,
                "active_subscriptions": list(connection.subscriptions)
            },
            timestamp=datetime.utcnow(),
            message_id=str(uuid.uuid4())
        )
        
        await self._send_message(connection_id, ack_message)
        
        logger.info(f"Client {connection_id} unsubscribed from {subscription_type.value}")
    
    async def broadcast_to_subscription(self, subscription_type: SubscriptionType, message: WebSocketMessage):
        """Broadcast message to all clients subscribed to a type"""
        subscribers = self.subscription_groups.get(subscription_type, set())
        
        if not subscribers:
            return
        
        # Send to all subscribers
        tasks = []
        for connection_id in subscribers.copy():  # Copy to avoid modification during iteration
            tasks.append(self._send_message(connection_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_client(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        await self._send_message(connection_id, message)
    
    async def _send_message(self, connection_id: str, message: WebSocketMessage):
        """Internal send message"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            # Add to message queue
            connection.message_queue.append(message)
            
            # Send message
            await connection.websocket.send_text(json.dumps(message.to_dict()))
            
        except WebSocketDisconnect:
            logger.info(f"Client {connection_id} disconnected during send")
            await self._disconnect_client(connection_id)
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            
            # Send error message
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                data={
                    "error": "Failed to send message",
                    "original_message_id": message.message_id
                },
                timestamp=datetime.utcnow(),
                message_id=str(uuid.uuid4())
            )
            
            try:
                await connection.websocket.send_text(json.dumps(error_message.to_dict()))
            except:
                await self._disconnect_client(connection_id)
    
    async def handle_client_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle message from client"""
        try:
            message_type = MessageType(message_data.get("type"))
            handler = self.message_handlers.get(message_type)
            
            if handler:
                await handler(connection_id, message_data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            
            # Send error response
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                data={
                    "error": "Failed to process message",
                    "details": str(e)
                },
                timestamp=datetime.utcnow(),
                message_id=str(uuid.uuid4())
            )
            
            await self._send_message(connection_id, error_message)
    
    async def _handle_heartbeat(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle heartbeat message"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.last_heartbeat = datetime.utcnow()
        
        # Send heartbeat response
        heartbeat_response = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={
                "server_time": datetime.utcnow().isoformat(),
                "connection_id": connection_id
            },
            timestamp=datetime.utcnow(),
            message_id=str(uuid.uuid4())
        )
        
        await self._send_message(connection_id, heartbeat_response)
    
    async def _handle_subscription_ack(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle subscription acknowledgment"""
        subscription_type = message_data.get("subscription_type")
        action = message_data.get("action", "subscribe")
        
        if subscription_type:
            try:
                sub_type = SubscriptionType(subscription_type)
                if action == "subscribe":
                    await self.subscribe_client(connection_id, sub_type)
                elif action == "unsubscribe":
                    await self.unsubscribe_client(connection_id, sub_type)
            except ValueError:
                logger.warning(f"Invalid subscription type: {subscription_type}")
    
    async def _handle_error(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle error message"""
        logger.error(f"Client {connection_id} reported error: {message_data}")
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
                
                # Check for timed out connections
                timed_out_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.last_heartbeat < timeout_threshold:
                        timed_out_connections.append(connection_id)
                
                # Disconnect timed out connections
                for connection_id in timed_out_connections:
                    logger.info(f"Connection {connection_id} timed out")
                    await self._disconnect_client(connection_id)
                
                # Send heartbeat to active connections
                heartbeat_message = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={
                        "server_time": current_time.isoformat(),
                        "connected_clients": len(self.connections)
                    },
                    timestamp=current_time,
                    message_id=str(uuid.uuid4())
                )
                
                # Send to all connections
                for connection_id in list(self.connections.keys()):
                    await self._send_message(connection_id, heartbeat_message)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _system_status_broadcaster(self):
        """Broadcast system status updates"""
        while self.running:
            try:
                # Get system status
                system_status = await system_monitor.get_system_overview()
                
                # Create message
                status_message = WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data=system_status.dict(),
                    timestamp=datetime.utcnow(),
                    message_id=str(uuid.uuid4())
                )
                
                # Broadcast to subscribers
                await self.broadcast_to_subscription(SubscriptionType.SYSTEM_STATUS, status_message)
                await self.broadcast_to_subscription(SubscriptionType.ALL, status_message)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system status broadcaster: {e}")
                await asyncio.sleep(5)
    
    async def _performance_metrics_broadcaster(self):
        """Broadcast performance metrics updates"""
        while self.running:
            try:
                # Get performance metrics
                performance_metrics = await system_monitor._get_system_performance_metrics()
                
                # Create message
                metrics_message = WebSocketMessage(
                    type=MessageType.PERFORMANCE_METRICS,
                    data=performance_metrics.dict(),
                    timestamp=datetime.utcnow(),
                    message_id=str(uuid.uuid4())
                )
                
                # Broadcast to subscribers
                await self.broadcast_to_subscription(SubscriptionType.PERFORMANCE_METRICS, metrics_message)
                await self.broadcast_to_subscription(SubscriptionType.ALL, metrics_message)
                
                await asyncio.sleep(3)  # Update every 3 seconds
                
            except Exception as e:
                logger.error(f"Error in performance metrics broadcaster: {e}")
                await asyncio.sleep(5)
    
    async def _alert_broadcaster(self):
        """Broadcast alert notifications"""
        while self.running:
            try:
                # Get active alerts
                alerts = await system_monitor.get_alerts()
                
                # Only send if there are active alerts
                if alerts:
                    # Create message
                    alert_message = WebSocketMessage(
                        type=MessageType.ALERT_NOTIFICATION,
                        data={
                            "alerts": [alert.dict() for alert in alerts],
                            "count": len(alerts)
                        },
                        timestamp=datetime.utcnow(),
                        message_id=str(uuid.uuid4())
                    )
                    
                    # Broadcast to subscribers
                    await self.broadcast_to_subscription(SubscriptionType.ALERTS, alert_message)
                    await self.broadcast_to_subscription(SubscriptionType.ALL, alert_message)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert broadcaster: {e}")
                await asyncio.sleep(5)
    
    async def _redis_subscriber(self):
        """Subscribe to Redis pub/sub for external events"""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("system_events", "component_events", "alert_events")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        
                        # Create WebSocket message
                        ws_message = WebSocketMessage(
                            type=MessageType(event_data.get("type", "system_status")),
                            data=event_data.get("data", {}),
                            timestamp=datetime.utcnow(),
                            message_id=str(uuid.uuid4())
                        )
                        
                        # Determine subscription type
                        if message["channel"] == "system_events":
                            await self.broadcast_to_subscription(SubscriptionType.SYSTEM_STATUS, ws_message)
                        elif message["channel"] == "component_events":
                            await self.broadcast_to_subscription(SubscriptionType.COMPONENT_HEALTH, ws_message)
                        elif message["channel"] == "alert_events":
                            await self.broadcast_to_subscription(SubscriptionType.ALERTS, ws_message)
                        
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Error in Redis subscriber: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        current_time = datetime.utcnow()
        
        stats = {
            "total_connections": len(self.connections),
            "active_connections": len([c for c in self.connections.values() if c.status == ConnectionStatus.CONNECTED]),
            "subscription_stats": {
                sub_type.value: len(subscribers) 
                for sub_type, subscribers in self.subscription_groups.items()
            },
            "uptime_seconds": int((current_time - datetime.utcnow()).total_seconds()),
            "connections_by_user": defaultdict(int)
        }
        
        # Count connections by user
        for connection in self.connections.values():
            stats["connections_by_user"][connection.user_info.username] += 1
        
        return stats
    
    def get_client_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get client information"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        
        return {
            "connection_id": connection_id,
            "user_id": connection.user_info.user_id,
            "username": connection.user_info.username,
            "connected_at": connection.connected_at.isoformat(),
            "last_heartbeat": connection.last_heartbeat.isoformat(),
            "subscriptions": list(connection.subscriptions),
            "status": connection.status.value,
            "message_queue_size": len(connection.message_queue)
        }

# Global real-time update manager
realtime_manager = RealTimeUpdateManager()

# WebSocket dependency
security = HTTPBearer()

async def get_websocket_user(websocket: WebSocket) -> UserInfo:
    """Get user info from WebSocket connection"""
    # Extract token from query parameters or headers
    token = websocket.query_params.get("token")
    
    if not token:
        # Check Authorization header
        auth_header = websocket.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")
    
    # Verify token
    try:
        from src.api.authentication import verify_token_payload
        credentials = type('Credentials', (), {'credentials': token})()
        payload = await verify_token_payload(token)
        
        # Get user info (simplified for WebSocket)
        from src.api.authentication import USERS_DB
        user_data = None
        for user in USERS_DB.values():
            if user["user_id"] == payload.user_id:
                user_data = user
                break
        
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
        
        return UserInfo(
            user_id=payload.user_id,
            username=payload.username,
            email=user_data["email"],
            role=user_data["role"],
            permissions=[],  # Simplified for WebSocket
            session_id=payload.session_id,
            login_time=datetime.fromtimestamp(payload.iat),
            last_activity=datetime.utcnow(),
            mfa_enabled=user_data.get("mfa_enabled", False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    connection_id = None
    
    try:
        # Accept connection
        await websocket.accept()
        
        # Get user info
        user_info = await get_websocket_user(websocket)
        
        # Connect client
        connection_id = await realtime_manager.connect_client(websocket, user_info)
        
        # Handle messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                message_data = json.loads(message)
                
                # Handle message
                await realtime_manager.handle_client_message(connection_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from client")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        
    finally:
        # Disconnect client
        if connection_id:
            await realtime_manager.disconnect_client(connection_id)

# API endpoints for WebSocket management
class WebSocketStatsResponse(BaseModel):
    """WebSocket statistics response"""
    total_connections: int
    active_connections: int
    subscription_stats: Dict[str, int]
    uptime_seconds: int
    connections_by_user: Dict[str, int]

class ClientInfoResponse(BaseModel):
    """Client information response"""
    connection_id: str
    user_id: str
    username: str
    connected_at: str
    last_heartbeat: str
    subscriptions: List[str]
    status: str
    message_queue_size: int

async def get_websocket_stats(user: UserInfo = Depends(verify_token)) -> WebSocketStatsResponse:
    """Get WebSocket statistics"""
    stats = realtime_manager.get_connection_stats()
    return WebSocketStatsResponse(**stats)

async def get_client_info(connection_id: str, user: UserInfo = Depends(verify_token)) -> ClientInfoResponse:
    """Get client information"""
    client_info = realtime_manager.get_client_info(connection_id)
    
    if not client_info:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return ClientInfoResponse(**client_info)

async def disconnect_client(connection_id: str, user: UserInfo = Depends(verify_token)):
    """Disconnect a client"""
    await realtime_manager.disconnect_client(connection_id)
    return {"message": "Client disconnected"}

# Startup and shutdown handlers
async def startup_realtime():
    """Start real-time update manager"""
    await realtime_manager.start()

async def shutdown_realtime():
    """Stop real-time update manager"""
    await realtime_manager.stop()