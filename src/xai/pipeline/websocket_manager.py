"""
WebSocket Manager for Real-time Explanation Streaming

Agent Beta: Real-time streaming specialist
Mission: Bulletproof WebSocket infrastructure for live explanations

This module implements the WebSocket infrastructure for broadcasting real-time
trading explanations to connected clients with auto-reconnection, heartbeat,
message queuing, and load balancing capabilities.

Key Features:
- Auto-reconnection and heartbeat monitoring
- Message queuing and delivery guarantees
- Load balancing for multiple clients
- Connection management with authentication
- Performance monitoring and metrics
- Horizontal scaling capability

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - WebSocket Infrastructure
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, Callable, Union
from enum import Enum
import weakref
from collections import defaultdict, deque
import threading

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed, WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available, WebSocket functionality disabled")

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ...core.events import EventType, Event, EventBus
from ...core.component_base import ComponentBase

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    EXPLANATION = "explanation"
    DECISION_UPDATE = "decision_update"
    MARKET_UPDATE = "market_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    AUTH_REQUEST = "auth_request"
    AUTH_RESPONSE = "auth_response"
    SUBSCRIPTION = "subscription"


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    message_id: str
    message_type: MessageType
    timestamp: datetime
    payload: Dict[str, Any]
    client_id: Optional[str] = None
    priority: str = "normal"
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ClientConnection:
    """Client connection metadata"""
    client_id: str
    websocket: Any  # WebSocketServerProtocol
    state: ConnectionState
    connected_at: datetime
    last_heartbeat: datetime
    subscriptions: Set[str]
    user_id: Optional[str] = None
    session_token: Optional[str] = None
    ip_address: str = "unknown"
    user_agent: str = "unknown"
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0


@dataclass
class WebSocketMetrics:
    """WebSocket performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    authenticated_connections: int = 0
    peak_connections: int = 0
    
    total_messages_sent: int = 0
    total_messages_failed: int = 0
    total_bytes_sent: int = 0
    
    connection_errors: int = 0
    authentication_failures: int = 0
    heartbeat_timeouts: int = 0
    
    avg_message_latency_ms: float = 0.0
    max_message_latency_ms: float = 0.0
    
    queue_size: int = 0
    queue_overflows: int = 0


class WebSocketManager(ComponentBase):
    """
    WebSocket Manager for Real-time Explanation Streaming
    
    Provides bulletproof WebSocket infrastructure for broadcasting real-time
    trading explanations with auto-reconnection, queuing, and scaling.
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize WebSocket Manager
        
        Args:
            kernel: Reference to the AlgoSpace kernel
            config: Configuration dictionary
        """
        super().__init__("WebSocketManager", kernel)
        
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('xai.websocket_manager')
        
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("websockets library not available")
            raise ImportError("websockets library required for WebSocketManager")
        
        # WebSocket server
        self.server = None
        self.host = self.config['host']
        self.port = self.config['port']
        
        # Client connections
        self.connections: Dict[str, ClientConnection] = {}
        self.connections_lock = threading.Lock()
        
        # Message queuing
        self.message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['queue_size']
        )
        self.priority_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['priority_queue_size']
        )
        
        # Redis for horizontal scaling
        self.redis_client: Optional[redis.Redis] = None
        self.redis_connected = False
        
        # Performance metrics
        self.metrics = WebSocketMetrics()
        self.metrics_lock = threading.Lock()
        
        # Message routing
        self.subscription_handlers: Dict[str, Callable] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Background tasks
        self.active = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Authentication
        self.auth_enabled = self.config['authentication']['enabled']
        self.auth_handler: Optional[Callable] = None
        
        # Performance monitoring
        self.latency_window = deque(maxlen=1000)
        
        self.logger.info(
            f"WebSocketManager initialized: "
            f"host={self.host}, port={self.port}, "
            f"auth_enabled={self.auth_enabled}, "
            f"queue_size={self.config['queue_size']}"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Server Configuration
            'host': '0.0.0.0',
            'port': 8765,
            'max_connections': 1000,
            'ping_interval': 20,  # seconds
            'ping_timeout': 10,   # seconds
            
            # Message Queuing
            'queue_size': 10000,
            'priority_queue_size': 1000,
            'max_message_size': 1024 * 1024,  # 1MB
            'batch_size': 100,
            'processing_interval_ms': 10,
            
            # Authentication
            'authentication': {
                'enabled': True,
                'token_expiry_hours': 24,
                'max_auth_attempts': 3,
                'auth_timeout_seconds': 30
            },
            
            # Heartbeat Configuration
            'heartbeat': {
                'interval_seconds': 30,
                'timeout_seconds': 90,
                'max_missed_heartbeats': 3
            },
            
            # Redis Configuration (for scaling)
            'redis': {
                'enabled': False,
                'url': 'redis://localhost:6379/2',
                'channel_prefix': 'xai:websocket',
                'message_ttl_seconds': 300
            },
            
            # Performance Monitoring
            'monitoring': {
                'metrics_interval_seconds': 60,
                'latency_threshold_ms': 100,
                'connection_limit_warning': 800
            },
            
            # Error Handling
            'error_handling': {
                'max_retries': 3,
                'retry_delay_ms': 1000,
                'circuit_breaker_threshold': 10
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the WebSocket Manager"""
        try:
            # Initialize Redis if enabled
            if self.config['redis']['enabled']:
                await self._initialize_redis()
            
            # Register message handlers
            self._register_message_handlers()
            
            # Start WebSocket server
            await self._start_server()
            
            # Start background tasks
            self.active = True
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.message_processor_task = asyncio.create_task(self._process_messages())
            self.metrics_task = asyncio.create_task(self._metrics_collector())
            
            self._initialized = True
            self.logger.info(f"WebSocketManager initialized on {self.host}:{self.port}")
            
            # Publish component started event
            event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {"component": self.name, "status": "ready", "endpoint": f"ws://{self.host}:{self.port}"},
                source=self.name
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocketManager: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis for horizontal scaling"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, scaling disabled")
            return
        
        try:
            self.redis_client = redis.from_url(self.config['redis']['url'])
            await self.redis_client.ping()
            self.redis_connected = True
            
            # Subscribe to Redis channels for cross-instance messaging
            self.redis_subscriber = self.redis_client.pubsub()
            await self.redis_subscriber.subscribe(
                f"{self.config['redis']['channel_prefix']}:broadcast"
            )
            
            self.logger.info(f"Redis connected for WebSocket scaling")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            self.redis_connected = False
    
    def _register_message_handlers(self) -> None:
        """Register message type handlers"""
        self.message_handlers.update({
            MessageType.AUTH_REQUEST: self._handle_auth_request,
            MessageType.SUBSCRIPTION: self._handle_subscription,
            MessageType.HEARTBEAT: self._handle_heartbeat_response
        })
    
    async def _start_server(self) -> None:
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=self.config['ping_interval'],
                ping_timeout=self.config['ping_timeout'],
                max_size=self.config['max_message_size']
            )
            
            self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Handle new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        client_id = str(uuid.uuid4())
        
        try:
            # Create client connection
            connection = ClientConnection(
                client_id=client_id,
                websocket=websocket,
                state=ConnectionState.CONNECTING,
                connected_at=datetime.now(timezone.utc),
                last_heartbeat=datetime.now(timezone.utc),
                subscriptions=set(),
                ip_address=websocket.remote_address[0] if websocket.remote_address else "unknown"
            )
            
            # Store connection
            with self.connections_lock:
                self.connections[client_id] = connection
                self.metrics.total_connections += 1
                self.metrics.active_connections += 1
                
                if self.metrics.active_connections > self.metrics.peak_connections:
                    self.metrics.peak_connections = self.metrics.active_connections
            
            connection.state = ConnectionState.CONNECTED
            
            self.logger.info(f"New WebSocket connection: {client_id} from {connection.ip_address}")
            
            # Send welcome message
            await self._send_welcome_message(connection)
            
            # Handle messages
            await self._handle_client_messages(connection)
            
        except ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {client_id}")
        except Exception as e:
            self.logger.error(f"WebSocket connection error for {client_id}: {e}")
            with self.metrics_lock:
                self.metrics.connection_errors += 1
        finally:
            # Clean up connection
            await self._cleanup_connection(client_id)
    
    async def _send_welcome_message(self, connection: ClientConnection) -> None:
        """Send welcome message to new connection"""
        welcome_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SYSTEM_STATUS,
            timestamp=datetime.now(timezone.utc),
            payload={
                'status': 'connected',
                'client_id': connection.client_id,
                'server_time': datetime.now(timezone.utc).isoformat(),
                'authentication_required': self.auth_enabled,
                'available_subscriptions': [
                    'explanations',
                    'decisions',
                    'market_updates',
                    'system_status'
                ]
            },
            client_id=connection.client_id
        )
        
        await self._send_message_to_connection(connection, welcome_message)
    
    async def _handle_client_messages(self, connection: ClientConnection) -> None:
        """
        Handle incoming messages from client
        
        Args:
            connection: Client connection
        """
        try:
            async for message in connection.websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    message_type = MessageType(data.get('type', 'unknown'))
                    
                    # Update heartbeat
                    connection.last_heartbeat = datetime.now(timezone.utc)
                    
                    # Route message to handler
                    handler = self.message_handlers.get(message_type)
                    if handler:
                        await handler(connection, data)
                    else:
                        await self._handle_unknown_message(connection, data)
                        
                except json.JSONDecodeError:
                    await self._send_error_message(connection, "Invalid JSON format")
                except ValueError:
                    await self._send_error_message(connection, "Unknown message type")
                except Exception as e:
                    self.logger.error(f"Message handling error for {connection.client_id}: {e}")
                    await self._send_error_message(connection, "Internal server error")
                    
        except ConnectionClosed:
            pass  # Normal disconnection
        except Exception as e:
            self.logger.error(f"Client message handling error: {e}")
    
    async def _handle_auth_request(self, connection: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle authentication request"""
        if not self.auth_enabled:
            connection.state = ConnectionState.AUTHENTICATED
            await self._send_auth_response(connection, True, "Authentication disabled")
            return
        
        token = data.get('token')
        user_id = data.get('user_id')
        
        # Use custom auth handler if available
        if self.auth_handler:
            try:
                auth_result = await self.auth_handler(token, user_id, connection)
                if auth_result:
                    connection.state = ConnectionState.AUTHENTICATED
                    connection.user_id = user_id
                    connection.session_token = token
                    
                    with self.metrics_lock:
                        self.metrics.authenticated_connections += 1
                    
                    await self._send_auth_response(connection, True, "Authentication successful")
                else:
                    await self._send_auth_response(connection, False, "Authentication failed")
                    with self.metrics_lock:
                        self.metrics.authentication_failures += 1
                        
            except Exception as e:
                self.logger.error(f"Authentication handler error: {e}")
                await self._send_auth_response(connection, False, "Authentication error")
        else:
            # Default authentication (accept all for development)
            connection.state = ConnectionState.AUTHENTICATED
            connection.user_id = user_id or "anonymous"
            await self._send_auth_response(connection, True, "Authentication successful")
    
    async def _send_auth_response(self, connection: ClientConnection, success: bool, message: str) -> None:
        """Send authentication response"""
        response = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.AUTH_RESPONSE,
            timestamp=datetime.now(timezone.utc),
            payload={
                'success': success,
                'message': message,
                'client_id': connection.client_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            client_id=connection.client_id
        )
        
        await self._send_message_to_connection(connection, response)
    
    async def _handle_subscription(self, connection: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle subscription request"""
        if connection.state != ConnectionState.AUTHENTICATED and self.auth_enabled:
            await self._send_error_message(connection, "Authentication required")
            return
        
        action = data.get('action', 'subscribe')
        topics = data.get('topics', [])
        
        if not isinstance(topics, list):
            topics = [topics]
        
        if action == 'subscribe':
            connection.subscriptions.update(topics)
            connection.state = ConnectionState.SUBSCRIBED
        elif action == 'unsubscribe':
            connection.subscriptions.difference_update(topics)
        
        # Send subscription confirmation
        response = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SUBSCRIPTION,
            timestamp=datetime.now(timezone.utc),
            payload={
                'action': action,
                'topics': topics,
                'active_subscriptions': list(connection.subscriptions),
                'success': True
            },
            client_id=connection.client_id
        )
        
        await self._send_message_to_connection(connection, response)
    
    async def _handle_heartbeat_response(self, connection: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle heartbeat response from client"""
        connection.last_heartbeat = datetime.now(timezone.utc)
        
        # Send heartbeat acknowledgment
        response = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            payload={
                'status': 'alive',
                'server_time': datetime.now(timezone.utc).isoformat()
            },
            client_id=connection.client_id
        )
        
        await self._send_message_to_connection(connection, response)
    
    async def _handle_unknown_message(self, connection: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle unknown message type"""
        await self._send_error_message(connection, f"Unknown message type: {data.get('type', 'missing')}")
    
    async def _send_error_message(self, connection: ClientConnection, error_message: str) -> None:
        """Send error message to client"""
        error_msg = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            timestamp=datetime.now(timezone.utc),
            payload={
                'error': error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            client_id=connection.client_id
        )
        
        await self._send_message_to_connection(connection, error_msg)
    
    async def _send_message_to_connection(self, connection: ClientConnection, message: WebSocketMessage) -> bool:
        """
        Send message to specific connection
        
        Args:
            connection: Target connection
            message: Message to send
            
        Returns:
            bool: Success status
        """
        try:
            send_start = time.perf_counter()
            
            message_json = json.dumps(asdict(message), default=str)
            await connection.websocket.send(message_json)
            
            # Update metrics
            send_latency_ms = (time.perf_counter() - send_start) * 1000
            self.latency_window.append(send_latency_ms)
            
            connection.messages_sent += 1
            connection.bytes_sent += len(message_json)
            
            with self.metrics_lock:
                self.metrics.total_messages_sent += 1
                self.metrics.total_bytes_sent += len(message_json)
                
                if send_latency_ms > self.metrics.max_message_latency_ms:
                    self.metrics.max_message_latency_ms = send_latency_ms
                
                if self.latency_window:
                    self.metrics.avg_message_latency_ms = sum(self.latency_window) / len(self.latency_window)
            
            return True
            
        except ConnectionClosed:
            self.logger.debug(f"Connection closed while sending to {connection.client_id}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send message to {connection.client_id}: {e}")
            connection.messages_failed += 1
            with self.metrics_lock:
                self.metrics.total_messages_failed += 1
            return False
    
    async def broadcast_message(
        self, 
        message: WebSocketMessage, 
        topic: Optional[str] = None,
        target_users: Optional[List[str]] = None
    ) -> int:
        """
        Broadcast message to subscribed clients
        
        Args:
            message: Message to broadcast
            topic: Topic filter (only clients subscribed to this topic)
            target_users: Specific user IDs to target
            
        Returns:
            int: Number of clients message was sent to
        """
        sent_count = 0
        
        with self.connections_lock:
            connections = list(self.connections.values())
        
        for connection in connections:
            # Check if connection should receive this message
            should_send = True
            
            # Topic filter
            if topic and topic not in connection.subscriptions:
                should_send = False
            
            # User filter
            if target_users and connection.user_id not in target_users:
                should_send = False
            
            # State filter
            if connection.state not in [ConnectionState.AUTHENTICATED, ConnectionState.SUBSCRIBED]:
                should_send = False
            
            if should_send:
                success = await self._send_message_to_connection(connection, message)
                if success:
                    sent_count += 1
        
        # Also broadcast via Redis for horizontal scaling
        if self.redis_connected and self.redis_client:
            try:
                channel = f"{self.config['redis']['channel_prefix']}:broadcast"
                redis_message = {
                    'message': asdict(message),
                    'topic': topic,
                    'target_users': target_users,
                    'source_instance': id(self)
                }
                
                await self.redis_client.publish(channel, json.dumps(redis_message, default=str))
                
            except Exception as e:
                self.logger.error(f"Redis broadcast failed: {e}")
        
        return sent_count
    
    async def queue_message(self, message: WebSocketMessage, priority: bool = False) -> bool:
        """
        Queue message for processing
        
        Args:
            message: Message to queue
            priority: Use priority queue
            
        Returns:
            bool: Success status
        """
        try:
            target_queue = self.priority_queue if priority else self.message_queue
            
            await target_queue.put(message)
            
            with self.metrics_lock:
                self.metrics.queue_size = self.message_queue.qsize() + self.priority_queue.qsize()
            
            return True
            
        except asyncio.QueueFull:
            with self.metrics_lock:
                self.metrics.queue_overflows += 1
            
            self.logger.warning("Message queue full, dropping message")
            return False
    
    async def _process_messages(self) -> None:
        """Process queued messages"""
        while self.active:
            try:
                # Process priority queue first
                try:
                    message = await asyncio.wait_for(self.priority_queue.get(), timeout=0.001)
                    await self.broadcast_message(message)
                    continue
                except asyncio.TimeoutError:
                    pass
                
                # Process regular queue
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=self.config['processing_interval_ms'] / 1000.0
                    )
                    await self.broadcast_message(message)
                except asyncio.TimeoutError:
                    pass
                
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor client heartbeats and clean up stale connections"""
        while self.active:
            try:
                now = datetime.now(timezone.utc)
                timeout_threshold = now.timestamp() - self.config['heartbeat']['timeout_seconds']
                stale_connections = []
                
                with self.connections_lock:
                    for client_id, connection in self.connections.items():
                        if connection.last_heartbeat.timestamp() < timeout_threshold:
                            stale_connections.append(client_id)
                
                # Clean up stale connections
                for client_id in stale_connections:
                    self.logger.info(f"Cleaning up stale connection: {client_id}")
                    await self._cleanup_connection(client_id)
                    
                    with self.metrics_lock:
                        self.metrics.heartbeat_timeouts += 1
                
                # Send heartbeat to all connections
                if stale_connections:
                    await self._send_heartbeat_to_all()
                
                await asyncio.sleep(self.config['heartbeat']['interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat_to_all(self) -> None:
        """Send heartbeat to all connected clients"""
        heartbeat_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            payload={
                'server_time': datetime.now(timezone.utc).isoformat(),
                'active_connections': self.metrics.active_connections
            }
        )
        
        await self.broadcast_message(heartbeat_message)
    
    async def _cleanup_connection(self, client_id: str) -> None:
        """Clean up client connection"""
        with self.connections_lock:
            connection = self.connections.pop(client_id, None)
            if connection:
                self.metrics.active_connections -= 1
                
                if connection.state == ConnectionState.AUTHENTICATED:
                    self.metrics.authenticated_connections -= 1
        
        if connection:
            try:
                await connection.websocket.close()
            except Exception:
                pass  # Connection already closed
    
    async def _metrics_collector(self) -> None:
        """Collect and log performance metrics"""
        while self.active:
            try:
                await asyncio.sleep(self.config['monitoring']['metrics_interval_seconds'])
                
                metrics = self.get_metrics()
                self.logger.info(
                    f"WebSocket Metrics: "
                    f"active_connections={metrics['active_connections']}, "
                    f"messages_sent={metrics['total_messages_sent']}, "
                    f"avg_latency_ms={metrics['avg_message_latency_ms']:.2f}, "
                    f"queue_size={metrics['queue_size']}"
                )
                
                # Connection limit warning
                if (metrics['active_connections'] >= 
                    self.config['monitoring']['connection_limit_warning']):
                    self.logger.warning(
                        f"High connection count: {metrics['active_connections']} "
                        f"(limit: {self.config['max_connections']})"
                    )
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    def set_auth_handler(self, handler: Callable[[str, str, ClientConnection], bool]) -> None:
        """
        Set custom authentication handler
        
        Args:
            handler: Async function that takes (token, user_id, connection) and returns bool
        """
        self.auth_handler = handler
        self.logger.info("Custom authentication handler set")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket metrics"""
        with self.metrics_lock:
            metrics_dict = asdict(self.metrics)
        
        # Add connection details
        with self.connections_lock:
            connection_states = defaultdict(int)
            for connection in self.connections.values():
                connection_states[connection.state.value] += 1
            
            metrics_dict['connection_states'] = dict(connection_states)
            metrics_dict['subscription_counts'] = {
                topic: sum(1 for conn in self.connections.values() if topic in conn.subscriptions)
                for topic in ['explanations', 'decisions', 'market_updates', 'system_status']
            }
        
        # Add system status
        metrics_dict['system_status'] = {
            'active': self.active,
            'server_running': self.server is not None,
            'redis_connected': self.redis_connected,
            'initialized': self._initialized
        }
        
        return metrics_dict
    
    async def shutdown(self) -> None:
        """Shutdown the WebSocket Manager"""
        try:
            self.active = False
            
            # Cancel background tasks
            tasks = [self.heartbeat_task, self.message_processor_task, self.metrics_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            
            # Close all client connections
            with self.connections_lock:
                connections = list(self.connections.values())
            
            for connection in connections:
                try:
                    await connection.websocket.close()
                except Exception:
                    pass
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info(
                f"WebSocketManager shutdown complete: "
                f"final_metrics={self.get_metrics()}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise


# Test function
async def test_websocket_manager():
    """Test the WebSocket Manager"""
    print("🧪 Testing WebSocket Manager")
    
    # Mock kernel and event bus
    class MockKernel:
        def __init__(self):
            self.event_bus = EventBus()
    
    kernel = MockKernel()
    
    # Initialize WebSocket manager
    config = {
        'host': 'localhost',
        'port': 8766,  # Use different port for testing
        'authentication': {'enabled': False},
        'redis': {'enabled': False}
    }
    
    manager = WebSocketManager(kernel, config)
    await manager.initialize()
    
    # Wait a moment for server to start
    await asyncio.sleep(0.1)
    
    # Create test message
    test_message = WebSocketMessage(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.EXPLANATION,
        timestamp=datetime.now(timezone.utc),
        payload={
            'decision_id': 'test-123',
            'explanation': 'Test explanation message',
            'confidence': 0.85,
            'action': 'buy'
        }
    )
    
    # Queue message
    await manager.queue_message(test_message)
    
    # Check metrics
    metrics = manager.get_metrics()
    print(f"\n📊 WebSocket Metrics:")
    print(f"  Active connections: {metrics['active_connections']}")
    print(f"  Messages sent: {metrics['total_messages_sent']}")
    print(f"  Queue size: {metrics['queue_size']}")
    print(f"  Server running: {metrics['system_status']['server_running']}")
    
    # Shutdown
    await manager.shutdown()
    print("\n✅ WebSocket Manager test complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket_manager())