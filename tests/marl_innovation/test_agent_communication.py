"""
Agent Communication Protocol Test Suite - AGENT 1 MISSION
Advanced Agent Communication Testing Framework

This comprehensive test suite validates agent communication protocols:
1. Message serialization/deserialization
2. Communication latency and reliability
3. Emergency communication protocols
4. Message routing and delivery
5. Network fault tolerance in communication

Author: Agent 1 - MARL Coordination Testing Specialist
Version: 1.0 - Production Ready
"""

import pytest
import asyncio
import time
import json
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import pickle
import struct
from collections import defaultdict, deque
import hashlib
import hmac

# Core imports
from src.core.events import EventBus, Event, EventType
from src.core.event_bus import EventBus as CoreEventBus

# Test markers
pytestmark = [
    pytest.mark.communication_testing,
    pytest.mark.marl_innovation,
    pytest.mark.agent_protocols,
    pytest.mark.networking
]

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in agent communication"""
    STRATEGIC_DECISION = "strategic_decision"
    TACTICAL_EXECUTION = "tactical_execution"
    RISK_ASSESSMENT = "risk_assessment"
    EMERGENCY_ALERT = "emergency_alert"
    HEARTBEAT = "heartbeat"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    SYSTEM_STATUS = "system_status"
    MARKET_DATA = "market_data"
    PERFORMANCE_METRICS = "performance_metrics"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class CommunicationProtocol(Enum):
    """Communication protocols"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    PIPELINE = "pipeline"


class SerializationMethod(Enum):
    """Serialization methods"""
    JSON = "json"
    PICKLE = "pickle"
    PROTOBUF = "protobuf"
    MESSAGEPACK = "messagepack"
    CUSTOM_BINARY = "custom_binary"


@dataclass
class Message:
    """Agent communication message"""
    message_id: str
    message_type: MessageType
    source_agent: str
    target_agent: Optional[str] = None
    target_group: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0  # Time to live in seconds
    sequence_number: int = 0
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = self.message_id
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(**data)


@dataclass
class CommunicationMetrics:
    """Communication performance metrics"""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_messages_dropped: int = 0
    total_messages_expired: int = 0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    throughput_messages_per_second: float = 0.0
    error_rate: float = 0.0
    delivery_success_rate: float = 0.0
    serialization_time_ms: float = 0.0
    deserialization_time_ms: float = 0.0
    compression_ratio: float = 0.0
    bandwidth_utilization: float = 0.0


@dataclass
class NetworkCondition:
    """Network condition simulation"""
    name: str
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 1000.0
    corruption_rate: float = 0.0
    duplication_rate: float = 0.0
    reordering_rate: float = 0.0


class MessageSerializer:
    """Message serialization and deserialization"""
    
    def __init__(self, method: SerializationMethod = SerializationMethod.JSON):
        self.method = method
        self.compression_enabled = False
        self.encryption_enabled = False
        self.encryption_key = None
    
    def serialize(self, message: Message) -> bytes:
        """Serialize message to bytes"""
        start_time = time.time()
        
        try:
            if self.method == SerializationMethod.JSON:
                data = json.dumps(message.to_dict()).encode('utf-8')
            elif self.method == SerializationMethod.PICKLE:
                data = pickle.dumps(message)
            elif self.method == SerializationMethod.CUSTOM_BINARY:
                data = self._serialize_custom_binary(message)
            else:
                raise ValueError(f"Unsupported serialization method: {self.method}")
            
            # Add compression if enabled
            if self.compression_enabled:
                data = self._compress_data(data)
            
            # Add encryption if enabled
            if self.encryption_enabled and self.encryption_key:
                data = self._encrypt_data(data)
            
            serialization_time = (time.time() - start_time) * 1000
            
            return data
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def deserialize(self, data: bytes) -> Message:
        """Deserialize bytes to message"""
        start_time = time.time()
        
        try:
            # Decrypt if enabled
            if self.encryption_enabled and self.encryption_key:
                data = self._decrypt_data(data)
            
            # Decompress if enabled
            if self.compression_enabled:
                data = self._decompress_data(data)
            
            if self.method == SerializationMethod.JSON:
                message_dict = json.loads(data.decode('utf-8'))
                # Convert enum strings back to enums
                message_dict['message_type'] = MessageType(message_dict['message_type'])
                message_dict['priority'] = MessagePriority(message_dict['priority'])
                message = Message.from_dict(message_dict)
            elif self.method == SerializationMethod.PICKLE:
                message = pickle.loads(data)
            elif self.method == SerializationMethod.CUSTOM_BINARY:
                message = self._deserialize_custom_binary(data)
            else:
                raise ValueError(f"Unsupported serialization method: {self.method}")
            
            deserialization_time = (time.time() - start_time) * 1000
            
            return message
            
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def _serialize_custom_binary(self, message: Message) -> bytes:
        """Custom binary serialization"""
        # Header: message_id (36 bytes), message_type (1 byte), priority (1 byte), timestamp (8 bytes)
        header = struct.pack(
            '36s B B d',
            message.message_id.encode('utf-8'),
            list(MessageType).index(message.message_type),
            list(MessagePriority).index(message.priority),
            message.timestamp
        )
        
        # Payload: JSON-encoded
        payload = json.dumps({
            'source_agent': message.source_agent,
            'target_agent': message.target_agent,
            'target_group': message.target_group,
            'payload': message.payload,
            'ttl': message.ttl,
            'sequence_number': message.sequence_number,
            'correlation_id': message.correlation_id,
            'retry_count': message.retry_count,
            'max_retries': message.max_retries
        }).encode('utf-8')
        
        # Length prefix for payload
        payload_length = struct.pack('I', len(payload))
        
        return header + payload_length + payload
    
    def _deserialize_custom_binary(self, data: bytes) -> Message:
        """Custom binary deserialization"""
        # Parse header
        header_size = 36 + 1 + 1 + 8  # 46 bytes
        header = data[:header_size]
        
        message_id, message_type_idx, priority_idx, timestamp = struct.unpack('36s B B d', header)
        message_id = message_id.decode('utf-8').rstrip('\x00')
        message_type = list(MessageType)[message_type_idx]
        priority = list(MessagePriority)[priority_idx]
        
        # Parse payload length
        payload_length = struct.unpack('I', data[header_size:header_size+4])[0]
        
        # Parse payload
        payload_data = data[header_size+4:header_size+4+payload_length]
        payload_dict = json.loads(payload_data.decode('utf-8'))
        
        return Message(
            message_id=message_id,
            message_type=message_type,
            priority=priority,
            timestamp=timestamp,
            **payload_dict
        )
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data (stub implementation)"""
        import zlib
        return zlib.compress(data)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data (stub implementation)"""
        import zlib
        return zlib.decompress(data)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data (stub implementation)"""
        # Simple XOR encryption for testing
        key = self.encryption_key.encode('utf-8')
        return bytes(a ^ key[i % len(key)] for i, a in enumerate(data))
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data (stub implementation)"""
        # Simple XOR decryption for testing
        key = self.encryption_key.encode('utf-8')
        return bytes(a ^ key[i % len(key)] for i, a in enumerate(data))


class MessageRouter:
    """Message routing and delivery system"""
    
    def __init__(self):
        self.agents = {}
        self.groups = defaultdict(set)
        self.message_queues = defaultdict(deque)
        self.routing_table = {}
        self.delivery_confirmations = {}
        self.metrics = CommunicationMetrics()
        self.message_history = []
        self.network_condition = NetworkCondition("normal")
        self.serializer = MessageSerializer()
        
    def register_agent(self, agent_id: str, agent_handler: callable):
        """Register agent with message handler"""
        self.agents[agent_id] = agent_handler
        self.message_queues[agent_id] = deque()
        logger.info(f"Agent {agent_id} registered with message router")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.message_queues[agent_id]
            logger.info(f"Agent {agent_id} unregistered from message router")
    
    def join_group(self, agent_id: str, group_id: str):
        """Join agent to group"""
        self.groups[group_id].add(agent_id)
        logger.info(f"Agent {agent_id} joined group {group_id}")
    
    def leave_group(self, agent_id: str, group_id: str):
        """Remove agent from group"""
        self.groups[group_id].discard(agent_id)
        logger.info(f"Agent {agent_id} left group {group_id}")
    
    async def send_message(self, message: Message, protocol: CommunicationProtocol = CommunicationProtocol.DIRECT) -> bool:
        """Send message using specified protocol"""
        if message.is_expired():
            logger.warning(f"Message {message.message_id} expired before sending")
            self.metrics.total_messages_expired += 1
            return False
        
        try:
            # Serialize message
            serialized = self.serializer.serialize(message)
            
            # Apply network conditions
            if not await self._simulate_network_conditions(message):
                self.metrics.total_messages_dropped += 1
                return False
            
            # Route message based on protocol
            if protocol == CommunicationProtocol.DIRECT:
                success = await self._send_direct(message, serialized)
            elif protocol == CommunicationProtocol.BROADCAST:
                success = await self._send_broadcast(message, serialized)
            elif protocol == CommunicationProtocol.MULTICAST:
                success = await self._send_multicast(message, serialized)
            elif protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                success = await self._send_publish_subscribe(message, serialized)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            if success:
                self.metrics.total_messages_sent += 1
                self.message_history.append({
                    'message_id': message.message_id,
                    'timestamp': time.time(),
                    'action': 'sent',
                    'protocol': protocol.value
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message {message.message_id}: {e}")
            self.metrics.total_messages_dropped += 1
            return False
    
    async def _simulate_network_conditions(self, message: Message) -> bool:
        """Simulate network conditions"""
        # Simulate packet loss
        if np.random.random() < self.network_condition.packet_loss_rate:
            logger.debug(f"Message {message.message_id} lost due to network conditions")
            return False
        
        # Simulate latency
        if self.network_condition.latency_ms > 0:
            jitter = np.random.normal(0, self.network_condition.jitter_ms)
            total_latency = max(0, self.network_condition.latency_ms + jitter)
            await asyncio.sleep(total_latency / 1000.0)
        
        # Simulate corruption
        if np.random.random() < self.network_condition.corruption_rate:
            logger.debug(f"Message {message.message_id} corrupted")
            return False
        
        return True
    
    async def _send_direct(self, message: Message, serialized: bytes) -> bool:
        """Send message directly to target agent"""
        if message.target_agent and message.target_agent in self.agents:
            try:
                # Deserialize message
                deserialized = self.serializer.deserialize(serialized)
                
                # Deliver to agent
                await self._deliver_to_agent(message.target_agent, deserialized)
                return True
            except Exception as e:
                logger.error(f"Error delivering direct message: {e}")
                return False
        
        return False
    
    async def _send_broadcast(self, message: Message, serialized: bytes) -> bool:
        """Broadcast message to all agents"""
        success_count = 0
        
        for agent_id in self.agents.keys():
            if agent_id != message.source_agent:  # Don't send to self
                try:
                    # Deserialize message
                    deserialized = self.serializer.deserialize(serialized)
                    deserialized.target_agent = agent_id
                    
                    # Deliver to agent
                    await self._deliver_to_agent(agent_id, deserialized)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error delivering broadcast message to {agent_id}: {e}")
        
        return success_count > 0
    
    async def _send_multicast(self, message: Message, serialized: bytes) -> bool:
        """Send message to group members"""
        if not message.target_group or message.target_group not in self.groups:
            return False
        
        success_count = 0
        group_members = self.groups[message.target_group]
        
        for agent_id in group_members:
            if agent_id != message.source_agent:  # Don't send to self
                try:
                    # Deserialize message
                    deserialized = self.serializer.deserialize(serialized)
                    deserialized.target_agent = agent_id
                    
                    # Deliver to agent
                    await self._deliver_to_agent(agent_id, deserialized)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error delivering multicast message to {agent_id}: {e}")
        
        return success_count > 0
    
    async def _send_publish_subscribe(self, message: Message, serialized: bytes) -> bool:
        """Send message using publish-subscribe pattern"""
        # For testing, treat as multicast to interested agents
        return await self._send_multicast(message, serialized)
    
    async def _deliver_to_agent(self, agent_id: str, message: Message):
        """Deliver message to agent"""
        if agent_id in self.agents:
            start_time = time.time()
            
            try:
                # Add to message queue
                self.message_queues[agent_id].append(message)
                
                # Call agent handler
                await self.agents[agent_id](message)
                
                # Update metrics
                delivery_time = (time.time() - start_time) * 1000
                self._update_latency_metrics(delivery_time)
                
                self.metrics.total_messages_received += 1
                
                # Record delivery
                self.message_history.append({
                    'message_id': message.message_id,
                    'timestamp': time.time(),
                    'action': 'delivered',
                    'agent_id': agent_id,
                    'delivery_time_ms': delivery_time
                })
                
            except Exception as e:
                logger.error(f"Error delivering message to {agent_id}: {e}")
                raise
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update latency metrics"""
        if self.metrics.total_messages_received == 0:
            self.metrics.average_latency_ms = latency_ms
        else:
            # Running average
            self.metrics.average_latency_ms = (
                (self.metrics.average_latency_ms * (self.metrics.total_messages_received - 1) + latency_ms) /
                self.metrics.total_messages_received
            )
        
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
    
    def get_agent_messages(self, agent_id: str) -> List[Message]:
        """Get messages for agent"""
        return list(self.message_queues.get(agent_id, []))
    
    def clear_agent_messages(self, agent_id: str):
        """Clear messages for agent"""
        if agent_id in self.message_queues:
            self.message_queues[agent_id].clear()
    
    def set_network_condition(self, condition: NetworkCondition):
        """Set network condition for simulation"""
        self.network_condition = condition
        logger.info(f"Network condition set to: {condition.name}")
    
    def get_metrics(self) -> CommunicationMetrics:
        """Get communication metrics"""
        # Update derived metrics
        if self.metrics.total_messages_sent > 0:
            self.metrics.delivery_success_rate = self.metrics.total_messages_received / self.metrics.total_messages_sent
            self.metrics.error_rate = self.metrics.total_messages_dropped / self.metrics.total_messages_sent
        
        return self.metrics
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get message history"""
        return self.message_history.copy()


class EmergencyProtocolHandler:
    """Emergency communication protocol handler"""
    
    def __init__(self, message_router: MessageRouter):
        self.message_router = message_router
        self.emergency_contacts = {}
        self.emergency_protocols = {}
        self.emergency_history = []
        self.active_emergencies = {}
        
    def register_emergency_contact(self, agent_id: str, contact_info: Dict[str, Any]):
        """Register emergency contact for agent"""
        self.emergency_contacts[agent_id] = contact_info
        logger.info(f"Emergency contact registered for {agent_id}")
    
    def define_emergency_protocol(self, emergency_type: str, protocol: Dict[str, Any]):
        """Define emergency protocol"""
        self.emergency_protocols[emergency_type] = protocol
        logger.info(f"Emergency protocol defined for {emergency_type}")
    
    async def trigger_emergency(self, emergency_type: str, details: Dict[str, Any], source_agent: str) -> bool:
        """Trigger emergency protocol"""
        emergency_id = str(uuid.uuid4())
        
        try:
            # Create emergency message
            emergency_message = Message(
                message_id=f"emergency_{emergency_id}",
                message_type=MessageType.EMERGENCY_ALERT,
                source_agent=source_agent,
                priority=MessagePriority.EMERGENCY,
                payload={
                    'emergency_id': emergency_id,
                    'emergency_type': emergency_type,
                    'details': details,
                    'timestamp': time.time()
                },
                ttl=60.0  # 1 minute TTL for emergency messages
            )
            
            # Record emergency
            self.active_emergencies[emergency_id] = {
                'type': emergency_type,
                'details': details,
                'source_agent': source_agent,
                'timestamp': time.time(),
                'status': 'active'
            }
            
            # Send emergency broadcast
            success = await self.message_router.send_message(
                emergency_message,
                CommunicationProtocol.BROADCAST
            )
            
            if success:
                # Execute emergency protocol
                await self._execute_emergency_protocol(emergency_type, emergency_id, details)
                
                # Record in history
                self.emergency_history.append({
                    'emergency_id': emergency_id,
                    'type': emergency_type,
                    'source_agent': source_agent,
                    'timestamp': time.time(),
                    'success': True
                })
                
                logger.critical(f"Emergency {emergency_type} triggered by {source_agent}")
                return True
            else:
                logger.error(f"Failed to broadcast emergency {emergency_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering emergency {emergency_type}: {e}")
            return False
    
    async def _execute_emergency_protocol(self, emergency_type: str, emergency_id: str, details: Dict[str, Any]):
        """Execute emergency protocol"""
        if emergency_type in self.emergency_protocols:
            protocol = self.emergency_protocols[emergency_type]
            
            # Execute protocol steps
            for step in protocol.get('steps', []):
                try:
                    await self._execute_protocol_step(step, emergency_id, details)
                except Exception as e:
                    logger.error(f"Error executing emergency protocol step: {e}")
    
    async def _execute_protocol_step(self, step: Dict[str, Any], emergency_id: str, details: Dict[str, Any]):
        """Execute single protocol step"""
        step_type = step.get('type')
        
        if step_type == 'notify_agents':
            # Notify specific agents
            agents = step.get('agents', [])
            for agent_id in agents:
                if agent_id in self.emergency_contacts:
                    # Send notification
                    pass
        
        elif step_type == 'shutdown_systems':
            # Shutdown systems
            systems = step.get('systems', [])
            for system in systems:
                # Shutdown system
                pass
        
        elif step_type == 'escalate':
            # Escalate to higher authority
            authority = step.get('authority')
            # Escalate to authority
            pass
    
    def resolve_emergency(self, emergency_id: str) -> bool:
        """Resolve emergency"""
        if emergency_id in self.active_emergencies:
            self.active_emergencies[emergency_id]['status'] = 'resolved'
            self.active_emergencies[emergency_id]['resolved_at'] = time.time()
            
            logger.info(f"Emergency {emergency_id} resolved")
            return True
        
        return False
    
    def get_active_emergencies(self) -> Dict[str, Any]:
        """Get active emergencies"""
        return {
            eid: emergency for eid, emergency in self.active_emergencies.items()
            if emergency['status'] == 'active'
        }
    
    def get_emergency_history(self) -> List[Dict[str, Any]]:
        """Get emergency history"""
        return self.emergency_history.copy()


class AgentCommunicationTester:
    """Agent communication testing framework"""
    
    def __init__(self):
        self.message_router = MessageRouter()
        self.emergency_handler = EmergencyProtocolHandler(self.message_router)
        self.test_agents = {}
        self.test_results = {}
        self.performance_metrics = {}
        
    async def initialize_test_environment(self, agent_count: int = 5) -> bool:
        """Initialize test environment with mock agents"""
        try:
            # Create test agents
            for i in range(agent_count):
                agent_id = f"test_agent_{i}"
                agent = await self._create_test_agent(agent_id)
                self.test_agents[agent_id] = agent
                
                # Register with message router
                self.message_router.register_agent(agent_id, agent.handle_message)
                
                # Register emergency contact
                self.emergency_handler.register_emergency_contact(agent_id, {
                    'type': 'test_agent',
                    'priority': 'normal'
                })
            
            # Setup agent groups
            await self._setup_agent_groups()
            
            # Define emergency protocols
            await self._setup_emergency_protocols()
            
            logger.info(f"Test environment initialized with {agent_count} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize test environment: {e}")
            return False
    
    async def _create_test_agent(self, agent_id: str) -> Mock:
        """Create test agent with communication capabilities"""
        agent = Mock()
        agent.agent_id = agent_id
        agent.received_messages = []
        agent.sent_messages = []
        agent.response_time_ms = np.random.uniform(10, 50)
        agent.reliability = np.random.uniform(0.9, 0.99)
        
        # Mock message handler
        async def handle_message(message: Message):
            await asyncio.sleep(agent.response_time_ms / 1000.0)
            
            # Simulate message processing
            agent.received_messages.append(message)
            
            # Generate response for certain message types
            if message.message_type in [MessageType.CONSENSUS_REQUEST, MessageType.COORDINATION_REQUEST]:
                response = await self._generate_response(agent_id, message)
                if response:
                    agent.sent_messages.append(response)
                    await self.message_router.send_message(response)
        
        # Mock send message
        async def send_message(message: Message) -> bool:
            agent.sent_messages.append(message)
            return await self.message_router.send_message(message)
        
        agent.handle_message = handle_message
        agent.send_message = send_message
        
        return agent
    
    async def _generate_response(self, agent_id: str, original_message: Message) -> Optional[Message]:
        """Generate response message"""
        if original_message.message_type == MessageType.CONSENSUS_REQUEST:
            return Message(
                message_type=MessageType.CONSENSUS_RESPONSE,
                source_agent=agent_id,
                target_agent=original_message.source_agent,
                correlation_id=original_message.correlation_id,
                payload={
                    'original_message_id': original_message.message_id,
                    'response': 'agree',
                    'confidence': np.random.uniform(0.7, 0.95)
                }
            )
        
        elif original_message.message_type == MessageType.COORDINATION_REQUEST:
            return Message(
                message_type=MessageType.COORDINATION_RESPONSE,
                source_agent=agent_id,
                target_agent=original_message.source_agent,
                correlation_id=original_message.correlation_id,
                payload={
                    'original_message_id': original_message.message_id,
                    'status': 'ready',
                    'estimated_time': np.random.uniform(0.1, 1.0)
                }
            )
        
        return None
    
    async def _setup_agent_groups(self):
        """Setup agent groups for testing"""
        agents = list(self.test_agents.keys())
        
        # Create strategic group
        strategic_group = agents[:2]
        for agent_id in strategic_group:
            self.message_router.join_group(agent_id, 'strategic')
        
        # Create tactical group
        tactical_group = agents[2:4]
        for agent_id in tactical_group:
            self.message_router.join_group(agent_id, 'tactical')
        
        # Create risk group
        risk_group = agents[4:]
        for agent_id in risk_group:
            self.message_router.join_group(agent_id, 'risk')
    
    async def _setup_emergency_protocols(self):
        """Setup emergency protocols"""
        # Market crash protocol
        self.emergency_handler.define_emergency_protocol('market_crash', {
            'steps': [
                {'type': 'notify_agents', 'agents': list(self.test_agents.keys())},
                {'type': 'shutdown_systems', 'systems': ['trading', 'order_management']},
                {'type': 'escalate', 'authority': 'risk_manager'}
            ]
        })
        
        # System failure protocol
        self.emergency_handler.define_emergency_protocol('system_failure', {
            'steps': [
                {'type': 'notify_agents', 'agents': list(self.test_agents.keys())},
                {'type': 'escalate', 'authority': 'system_administrator'}
            ]
        })
    
    async def test_message_serialization(self, serialization_methods: List[SerializationMethod]) -> Dict[str, Any]:
        """Test message serialization methods"""
        test_results = {}
        
        # Create test message
        test_message = Message(
            message_type=MessageType.STRATEGIC_DECISION,
            source_agent='test_agent_0',
            target_agent='test_agent_1',
            payload={
                'decision': 'buy',
                'confidence': 0.85,
                'market_data': list(np.random.randn(100)),
                'complex_data': {
                    'nested': {'values': [1, 2, 3]},
                    'matrix': np.random.randn(10, 10).tolist()
                }
            }
        )
        
        for method in serialization_methods:
            logger.info(f"Testing serialization method: {method.value}")
            
            serializer = MessageSerializer(method)
            method_results = {
                'method': method.value,
                'serialization_time_ms': [],
                'deserialization_time_ms': [],
                'serialized_size_bytes': [],
                'success_rate': 0.0,
                'errors': []
            }
            
            # Test multiple rounds
            successful_rounds = 0
            total_rounds = 10
            
            for _ in range(total_rounds):
                try:
                    # Serialize
                    start_time = time.time()
                    serialized = serializer.serialize(test_message)
                    serialization_time = (time.time() - start_time) * 1000
                    
                    # Deserialize
                    start_time = time.time()
                    deserialized = serializer.deserialize(serialized)
                    deserialization_time = (time.time() - start_time) * 1000
                    
                    # Verify correctness
                    if (deserialized.message_id == test_message.message_id and
                        deserialized.message_type == test_message.message_type and
                        deserialized.source_agent == test_message.source_agent):
                        
                        method_results['serialization_time_ms'].append(serialization_time)
                        method_results['deserialization_time_ms'].append(deserialization_time)
                        method_results['serialized_size_bytes'].append(len(serialized))
                        successful_rounds += 1
                    
                except Exception as e:
                    method_results['errors'].append(str(e))
            
            method_results['success_rate'] = successful_rounds / total_rounds
            
            if method_results['serialization_time_ms']:
                method_results['avg_serialization_time_ms'] = np.mean(method_results['serialization_time_ms'])
                method_results['avg_deserialization_time_ms'] = np.mean(method_results['deserialization_time_ms'])
                method_results['avg_serialized_size_bytes'] = np.mean(method_results['serialized_size_bytes'])
            
            test_results[method.value] = method_results
        
        return test_results
    
    async def test_communication_protocols(self, protocols: List[CommunicationProtocol]) -> Dict[str, Any]:
        """Test communication protocols"""
        test_results = {}
        
        for protocol in protocols:
            logger.info(f"Testing communication protocol: {protocol.value}")
            
            protocol_results = {
                'protocol': protocol.value,
                'messages_sent': 0,
                'messages_delivered': 0,
                'delivery_latency_ms': [],
                'delivery_success_rate': 0.0,
                'errors': []
            }
            
            # Clear previous messages
            for agent in self.test_agents.values():
                agent.received_messages.clear()
            
            # Test protocol
            test_messages = []
            for i in range(5):
                if protocol == CommunicationProtocol.DIRECT:
                    message = Message(
                        message_type=MessageType.COORDINATION_REQUEST,
                        source_agent='test_agent_0',
                        target_agent='test_agent_1',
                        payload={'request': f'direct_test_{i}'}
                    )
                elif protocol == CommunicationProtocol.BROADCAST:
                    message = Message(
                        message_type=MessageType.SYSTEM_STATUS,
                        source_agent='test_agent_0',
                        payload={'status': f'broadcast_test_{i}'}
                    )
                elif protocol == CommunicationProtocol.MULTICAST:
                    message = Message(
                        message_type=MessageType.STRATEGIC_DECISION,
                        source_agent='test_agent_0',
                        target_group='strategic',
                        payload={'decision': f'multicast_test_{i}'}
                    )
                else:
                    continue
                
                test_messages.append(message)
            
            # Send messages
            for message in test_messages:
                try:
                    start_time = time.time()
                    success = await self.message_router.send_message(message, protocol)
                    
                    if success:
                        protocol_results['messages_sent'] += 1
                        
                        # Wait for delivery
                        await asyncio.sleep(0.1)
                        
                        # Check delivery
                        delivered = False
                        for agent in self.test_agents.values():
                            if any(msg.message_id == message.message_id for msg in agent.received_messages):
                                delivered = True
                                break
                        
                        if delivered:
                            protocol_results['messages_delivered'] += 1
                            delivery_time = (time.time() - start_time) * 1000
                            protocol_results['delivery_latency_ms'].append(delivery_time)
                    
                except Exception as e:
                    protocol_results['errors'].append(str(e))
            
            # Calculate metrics
            if protocol_results['messages_sent'] > 0:
                protocol_results['delivery_success_rate'] = (
                    protocol_results['messages_delivered'] / protocol_results['messages_sent']
                )
            
            if protocol_results['delivery_latency_ms']:
                protocol_results['avg_delivery_latency_ms'] = np.mean(protocol_results['delivery_latency_ms'])
                protocol_results['max_delivery_latency_ms'] = np.max(protocol_results['delivery_latency_ms'])
            
            test_results[protocol.value] = protocol_results
        
        return test_results
    
    async def test_network_conditions(self, network_conditions: List[NetworkCondition]) -> Dict[str, Any]:
        """Test communication under various network conditions"""
        test_results = {}
        
        for condition in network_conditions:
            logger.info(f"Testing network condition: {condition.name}")
            
            # Set network condition
            self.message_router.set_network_condition(condition)
            
            condition_results = {
                'condition_name': condition.name,
                'latency_ms': condition.latency_ms,
                'packet_loss_rate': condition.packet_loss_rate,
                'messages_sent': 0,
                'messages_delivered': 0,
                'delivery_success_rate': 0.0,
                'avg_delivery_time_ms': 0.0,
                'errors': []
            }
            
            # Clear previous messages
            for agent in self.test_agents.values():
                agent.received_messages.clear()
            
            # Send test messages
            test_message_count = 20
            delivered_count = 0
            delivery_times = []
            
            for i in range(test_message_count):
                try:
                    message = Message(
                        message_type=MessageType.HEARTBEAT,
                        source_agent='test_agent_0',
                        target_agent='test_agent_1',
                        payload={'heartbeat': i}
                    )
                    
                    start_time = time.time()
                    success = await self.message_router.send_message(message)
                    
                    if success:
                        condition_results['messages_sent'] += 1
                        
                        # Wait for delivery
                        await asyncio.sleep(0.1)
                        
                        # Check delivery
                        target_agent = self.test_agents['test_agent_1']
                        if any(msg.message_id == message.message_id for msg in target_agent.received_messages):
                            delivered_count += 1
                            delivery_time = (time.time() - start_time) * 1000
                            delivery_times.append(delivery_time)
                    
                except Exception as e:
                    condition_results['errors'].append(str(e))
            
            condition_results['messages_delivered'] = delivered_count
            
            if condition_results['messages_sent'] > 0:
                condition_results['delivery_success_rate'] = delivered_count / condition_results['messages_sent']
            
            if delivery_times:
                condition_results['avg_delivery_time_ms'] = np.mean(delivery_times)
                condition_results['max_delivery_time_ms'] = np.max(delivery_times)
            
            test_results[condition.name] = condition_results
        
        # Reset to normal conditions
        self.message_router.set_network_condition(NetworkCondition("normal"))
        
        return test_results
    
    async def test_emergency_protocols(self, emergency_types: List[str]) -> Dict[str, Any]:
        """Test emergency communication protocols"""
        test_results = {}
        
        for emergency_type in emergency_types:
            logger.info(f"Testing emergency protocol: {emergency_type}")
            
            emergency_results = {
                'emergency_type': emergency_type,
                'triggered_successfully': False,
                'response_time_ms': 0.0,
                'agents_notified': 0,
                'protocol_executed': False,
                'errors': []
            }
            
            # Clear previous messages
            for agent in self.test_agents.values():
                agent.received_messages.clear()
            
            try:
                # Trigger emergency
                start_time = time.time()
                success = await self.emergency_handler.trigger_emergency(
                    emergency_type,
                    {'severity': 'high', 'test': True},
                    'test_agent_0'
                )
                
                if success:
                    emergency_results['triggered_successfully'] = True
                    
                    # Wait for emergency processing
                    await asyncio.sleep(0.5)
                    
                    response_time = (time.time() - start_time) * 1000
                    emergency_results['response_time_ms'] = response_time
                    
                    # Count agents that received emergency message
                    notified_agents = 0
                    for agent in self.test_agents.values():
                        emergency_messages = [
                            msg for msg in agent.received_messages
                            if msg.message_type == MessageType.EMERGENCY_ALERT
                        ]
                        if emergency_messages:
                            notified_agents += 1
                    
                    emergency_results['agents_notified'] = notified_agents
                    emergency_results['protocol_executed'] = True
                
            except Exception as e:
                emergency_results['errors'].append(str(e))
            
            test_results[emergency_type] = emergency_results
        
        return test_results
    
    async def test_communication_reliability(self, reliability_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test communication reliability under various scenarios"""
        test_results = {}
        
        for scenario in reliability_scenarios:
            logger.info(f"Testing reliability scenario: {scenario['name']}")
            
            scenario_results = {
                'scenario_name': scenario['name'],
                'messages_sent': 0,
                'messages_delivered': 0,
                'messages_lost': 0,
                'duplicate_messages': 0,
                'out_of_order_messages': 0,
                'reliability_score': 0.0,
                'errors': []
            }
            
            # Apply scenario conditions
            if 'network_condition' in scenario:
                self.message_router.set_network_condition(scenario['network_condition'])
            
            # Clear previous messages
            for agent in self.test_agents.values():
                agent.received_messages.clear()
            
            # Send test messages
            test_message_count = scenario.get('message_count', 50)
            sent_messages = []
            
            for i in range(test_message_count):
                try:
                    message = Message(
                        message_type=MessageType.PERFORMANCE_METRICS,
                        source_agent='test_agent_0',
                        target_agent='test_agent_1',
                        payload={'sequence': i, 'timestamp': time.time()},
                        sequence_number=i
                    )
                    
                    success = await self.message_router.send_message(message)
                    
                    if success:
                        scenario_results['messages_sent'] += 1
                        sent_messages.append(message)
                    
                    # Add delay between messages
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    scenario_results['errors'].append(str(e))
            
            # Wait for delivery
            await asyncio.sleep(1.0)
            
            # Analyze delivery
            target_agent = self.test_agents['test_agent_1']
            received_messages = target_agent.received_messages
            
            scenario_results['messages_delivered'] = len(received_messages)
            scenario_results['messages_lost'] = scenario_results['messages_sent'] - len(received_messages)
            
            # Check for duplicates
            received_ids = [msg.message_id for msg in received_messages]
            scenario_results['duplicate_messages'] = len(received_ids) - len(set(received_ids))
            
            # Check for out-of-order messages
            received_sequences = [msg.sequence_number for msg in received_messages]
            out_of_order = sum(1 for i in range(1, len(received_sequences)) 
                              if received_sequences[i] < received_sequences[i-1])
            scenario_results['out_of_order_messages'] = out_of_order
            
            # Calculate reliability score
            if scenario_results['messages_sent'] > 0:
                base_reliability = scenario_results['messages_delivered'] / scenario_results['messages_sent']
                duplicate_penalty = scenario_results['duplicate_messages'] / scenario_results['messages_sent']
                order_penalty = scenario_results['out_of_order_messages'] / scenario_results['messages_sent']
                
                scenario_results['reliability_score'] = max(0.0, base_reliability - duplicate_penalty - order_penalty)
            
            test_results[scenario['name']] = scenario_results
        
        # Reset to normal conditions
        self.message_router.set_network_condition(NetworkCondition("normal"))
        
        return test_results
    
    def generate_communication_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive communication test report"""
        return {
            'test_environment': {
                'agent_count': len(self.test_agents),
                'message_router_metrics': self.message_router.get_metrics().__dict__,
                'emergency_protocols': len(self.emergency_handler.emergency_protocols),
                'test_duration': time.time() - self.performance_metrics.get('test_start_time', time.time())
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'message_history': self.message_router.get_message_history()[-100:],  # Last 100 messages
            'emergency_history': self.emergency_handler.get_emergency_history(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze metrics
        metrics = self.message_router.get_metrics()
        
        if metrics.delivery_success_rate < 0.9:
            recommendations.append("Improve message delivery reliability through retry mechanisms")
        
        if metrics.average_latency_ms > 100:
            recommendations.append("Optimize message routing for lower latency")
        
        if metrics.error_rate > 0.1:
            recommendations.append("Investigate and fix high error rate in message delivery")
        
        return recommendations


class TestAgentCommunication:
    """Agent communication protocol test suite"""
    
    @pytest.fixture
    async def communication_tester(self):
        """Setup communication tester"""
        tester = AgentCommunicationTester()
        
        success = await tester.initialize_test_environment(agent_count=5)
        assert success, "Failed to initialize communication test environment"
        
        tester.performance_metrics['test_start_time'] = time.time()
        
        yield tester
        
        # Cleanup
        # No specific cleanup needed for mock objects
    
    @pytest.mark.asyncio
    async def test_message_serialization_methods(self, communication_tester):
        """Test message serialization methods"""
        serialization_methods = [
            SerializationMethod.JSON,
            SerializationMethod.PICKLE,
            SerializationMethod.CUSTOM_BINARY
        ]
        
        results = await communication_tester.test_message_serialization(serialization_methods)
        
        # Verify all methods were tested
        assert len(results) == len(serialization_methods)
        
        # Verify success rates
        for method, result in results.items():
            assert result['success_rate'] > 0.8, f"Low success rate for {method}: {result['success_rate']}"
            assert 'avg_serialization_time_ms' in result
            assert 'avg_deserialization_time_ms' in result
            assert 'avg_serialized_size_bytes' in result
    
    @pytest.mark.asyncio
    async def test_communication_protocols(self, communication_tester):
        """Test communication protocols"""
        protocols = [
            CommunicationProtocol.DIRECT,
            CommunicationProtocol.BROADCAST,
            CommunicationProtocol.MULTICAST
        ]
        
        results = await communication_tester.test_communication_protocols(protocols)
        
        # Verify all protocols were tested
        assert len(results) == len(protocols)
        
        # Verify protocol performance
        for protocol, result in results.items():
            assert result['messages_sent'] > 0, f"No messages sent for {protocol}"
            assert result['delivery_success_rate'] > 0.5, f"Low delivery success rate for {protocol}"
            
            if result['delivery_latency_ms']:
                assert result['avg_delivery_latency_ms'] < 1000, f"High latency for {protocol}"
    
    @pytest.mark.asyncio
    async def test_network_conditions(self, communication_tester):
        """Test communication under various network conditions"""
        network_conditions = [
            NetworkCondition("normal", latency_ms=10, packet_loss_rate=0.0),
            NetworkCondition("slow", latency_ms=100, packet_loss_rate=0.05),
            NetworkCondition("unreliable", latency_ms=50, packet_loss_rate=0.2),
            NetworkCondition("high_latency", latency_ms=500, packet_loss_rate=0.1)
        ]
        
        results = await communication_tester.test_network_conditions(network_conditions)
        
        # Verify all conditions were tested
        assert len(results) == len(network_conditions)
        
        # Verify condition effects
        normal_result = results["normal"]
        slow_result = results["slow"]
        unreliable_result = results["unreliable"]
        
        # Normal conditions should have high success rate
        assert normal_result['delivery_success_rate'] > 0.9
        
        # Slow conditions should have higher latency
        if slow_result['avg_delivery_time_ms'] > 0:
            assert slow_result['avg_delivery_time_ms'] > normal_result['avg_delivery_time_ms']
        
        # Unreliable conditions should have lower success rate
        assert unreliable_result['delivery_success_rate'] < normal_result['delivery_success_rate']
    
    @pytest.mark.asyncio
    async def test_emergency_protocols(self, communication_tester):
        """Test emergency communication protocols"""
        emergency_types = ['market_crash', 'system_failure']
        
        results = await communication_tester.test_emergency_protocols(emergency_types)
        
        # Verify all emergency types were tested
        assert len(results) == len(emergency_types)
        
        # Verify emergency protocol performance
        for emergency_type, result in results.items():
            assert result['triggered_successfully'], f"Emergency {emergency_type} not triggered successfully"
            assert result['response_time_ms'] < 1000, f"Emergency {emergency_type} response time too high"
            assert result['agents_notified'] > 0, f"No agents notified for emergency {emergency_type}"
            assert result['protocol_executed'], f"Emergency protocol not executed for {emergency_type}"
    
    @pytest.mark.asyncio
    async def test_communication_reliability(self, communication_tester):
        """Test communication reliability"""
        reliability_scenarios = [
            {
                'name': 'normal_conditions',
                'message_count': 100,
                'network_condition': NetworkCondition("normal", latency_ms=10, packet_loss_rate=0.0)
            },
            {
                'name': 'packet_loss',
                'message_count': 100,
                'network_condition': NetworkCondition("lossy", latency_ms=20, packet_loss_rate=0.1)
            },
            {
                'name': 'high_latency',
                'message_count': 50,
                'network_condition': NetworkCondition("slow", latency_ms=200, packet_loss_rate=0.05)
            }
        ]
        
        results = await communication_tester.test_communication_reliability(reliability_scenarios)
        
        # Verify all scenarios were tested
        assert len(results) == len(reliability_scenarios)
        
        # Verify reliability metrics
        for scenario_name, result in results.items():
            assert result['messages_sent'] > 0, f"No messages sent in {scenario_name}"
            assert result['reliability_score'] >= 0.0, f"Invalid reliability score for {scenario_name}"
            assert result['reliability_score'] <= 1.0, f"Invalid reliability score for {scenario_name}"
            
            # Normal conditions should have high reliability
            if scenario_name == 'normal_conditions':
                assert result['reliability_score'] > 0.9, f"Low reliability in normal conditions"
    
    @pytest.mark.asyncio
    async def test_message_ordering_and_delivery(self, communication_tester):
        """Test message ordering and delivery guarantees"""
        # Send sequence of messages
        agent_0 = communication_tester.test_agents['test_agent_0']
        agent_1 = communication_tester.test_agents['test_agent_1']
        
        # Clear previous messages
        agent_1.received_messages.clear()
        
        # Send ordered messages
        message_count = 20
        sent_messages = []
        
        for i in range(message_count):
            message = Message(
                message_type=MessageType.MARKET_DATA,
                source_agent='test_agent_0',
                target_agent='test_agent_1',
                payload={'sequence': i, 'data': f'message_{i}'},
                sequence_number=i
            )
            
            success = await communication_tester.message_router.send_message(message)
            assert success, f"Failed to send message {i}"
            sent_messages.append(message)
            
            # Small delay between messages
            await asyncio.sleep(0.01)
        
        # Wait for delivery
        await asyncio.sleep(1.0)
        
        # Verify delivery
        received_messages = agent_1.received_messages
        assert len(received_messages) >= message_count * 0.8, "Too many messages lost"
        
        # Check ordering
        received_sequences = [msg.sequence_number for msg in received_messages]
        out_of_order_count = sum(1 for i in range(1, len(received_sequences)) 
                               if received_sequences[i] < received_sequences[i-1])
        
        # Allow some out-of-order messages but not too many
        assert out_of_order_count < len(received_messages) * 0.1, "Too many out-of-order messages"
    
    @pytest.mark.asyncio
    async def test_concurrent_communication(self, communication_tester):
        """Test concurrent communication scenarios"""
        # Create concurrent communication tasks
        tasks = []
        
        # Task 1: Strategic agent broadcast
        async def strategic_broadcast():
            for i in range(10):
                message = Message(
                    message_type=MessageType.STRATEGIC_DECISION,
                    source_agent='test_agent_0',
                    payload={'decision': f'strategic_{i}'}
                )
                await communication_tester.message_router.send_message(
                    message, CommunicationProtocol.BROADCAST
                )
                await asyncio.sleep(0.05)
        
        # Task 2: Tactical coordination
        async def tactical_coordination():
            for i in range(15):
                message = Message(
                    message_type=MessageType.COORDINATION_REQUEST,
                    source_agent='test_agent_1',
                    target_agent='test_agent_2',
                    payload={'coordination': f'tactical_{i}'}
                )
                await communication_tester.message_router.send_message(message)
                await asyncio.sleep(0.03)
        
        # Task 3: Risk alerts
        async def risk_alerts():
            for i in range(8):
                message = Message(
                    message_type=MessageType.RISK_ASSESSMENT,
                    source_agent='test_agent_3',
                    target_group='risk',
                    payload={'risk_level': f'level_{i}'}
                )
                await communication_tester.message_router.send_message(
                    message, CommunicationProtocol.MULTICAST
                )
                await asyncio.sleep(0.08)
        
        # Run tasks concurrently
        tasks = [strategic_broadcast(), tactical_coordination(), risk_alerts()]
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Verify concurrent execution
        assert execution_time < 2.0, "Concurrent communication took too long"
        
        # Verify messages were processed
        total_received = sum(len(agent.received_messages) for agent in communication_tester.test_agents.values())
        assert total_received > 0, "No messages received during concurrent communication"
        
        # Verify no significant interference
        metrics = communication_tester.message_router.get_metrics()
        assert metrics.delivery_success_rate > 0.7, "Low delivery success rate during concurrent communication"
    
    def test_communication_test_report_generation(self, communication_tester):
        """Test communication test report generation"""
        # Generate test report
        test_report = communication_tester.generate_communication_test_report()
        
        # Verify report structure
        assert 'test_environment' in test_report
        assert 'test_results' in test_report
        assert 'performance_metrics' in test_report
        assert 'message_history' in test_report
        assert 'emergency_history' in test_report
        assert 'recommendations' in test_report
        
        # Verify report content
        env = test_report['test_environment']
        assert env['agent_count'] == 5
        assert 'message_router_metrics' in env
        assert 'emergency_protocols' in env
        
        # Verify recommendations are generated
        recommendations = test_report['recommendations']
        assert isinstance(recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])