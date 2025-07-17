#!/usr/bin/env python3
"""
Redis State Synchronization for Trading Engine
AGENT 2: Trading Engine RTO Specialist

This module implements real-time state synchronization between active and passive
trading engine instances using Redis as the backbone. Critical for achieving
<5s RTO through hot standby capabilities.

Key Features:
- Real-time state synchronization (500ms intervals)
- Conflict resolution with vector clocks
- Atomic state updates with transactions
- Automatic failover state management
- Performance monitoring and alerting
- State compression and serialization
"""

import asyncio
import time
import json
import pickle
import gzip
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import defaultdict

import redis.asyncio as redis
import redis.exceptions
from pydantic import BaseModel, Field

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, Event, EventType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InstanceRole(Enum):
    """Trading engine instance roles"""
    ACTIVE = "active"
    PASSIVE = "passive"
    UNKNOWN = "unknown"

class StateType(Enum):
    """Types of state that can be synchronized"""
    TRADING_STATE = "trading_state"
    PORTFOLIO_STATE = "portfolio_state"
    RISK_STATE = "risk_state"
    EXECUTION_STATE = "execution_state"
    CIRCUIT_BREAKER_STATE = "circuit_breaker_state"
    HEALTH_STATE = "health_state"

@dataclass
class VectorClock:
    """Vector clock for conflict resolution"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def tick(self, node_id: str):
        """Increment clock for node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update with another vector clock"""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)
    
    def compare(self, other: 'VectorClock') -> str:
        """Compare with another vector clock"""
        self_nodes = set(self.clocks.keys())
        other_nodes = set(other.clocks.keys())
        all_nodes = self_nodes | other_nodes
        
        self_greater = False
        other_greater = False
        
        for node_id in all_nodes:
            self_clock = self.clocks.get(node_id, 0)
            other_clock = other.clocks.get(node_id, 0)
            
            if self_clock > other_clock:
                self_greater = True
            elif self_clock < other_clock:
                other_greater = True
        
        if self_greater and not other_greater:
            return "greater"
        elif other_greater and not self_greater:
            return "less"
        elif not self_greater and not other_greater:
            return "equal"
        else:
            return "concurrent"

@dataclass
class StateSnapshot:
    """State snapshot with metadata"""
    state_type: StateType
    data: Dict[str, Any]
    timestamp: float
    instance_id: str
    vector_clock: VectorClock
    version: int
    checksum: str
    compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'state_type': self.state_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'instance_id': self.instance_id,
            'vector_clock': asdict(self.vector_clock),
            'version': self.version,
            'checksum': self.checksum,
            'compressed': self.compressed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create from dictionary"""
        return cls(
            state_type=StateType(data['state_type']),
            data=data['data'],
            timestamp=data['timestamp'],
            instance_id=data['instance_id'],
            vector_clock=VectorClock(**data['vector_clock']),
            version=data['version'],
            checksum=data['checksum'],
            compressed=data.get('compressed', False)
        )

@dataclass
class SyncMetrics:
    """Synchronization performance metrics"""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_resolved: int = 0
    avg_sync_time_ms: float = 0.0
    max_sync_time_ms: float = 0.0
    compression_ratio: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    last_sync_time: float = 0.0
    
    def update_sync_time(self, sync_time_ms: float):
        """Update synchronization time metrics"""
        self.total_syncs += 1
        self.avg_sync_time_ms = (self.avg_sync_time_ms * (self.total_syncs - 1) + sync_time_ms) / self.total_syncs
        self.max_sync_time_ms = max(self.max_sync_time_ms, sync_time_ms)
        self.last_sync_time = time.time()

class StateSerializer:
    """High-performance state serialization"""
    
    @staticmethod
    def serialize(data: Any, compress: bool = True) -> bytes:
        """Serialize data with optional compression"""
        # Use pickle for speed
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        if compress and len(serialized) > 1024:  # Compress only if larger than 1KB
            serialized = gzip.compress(serialized)
            return serialized
        
        return serialized
    
    @staticmethod
    def deserialize(data: bytes, compressed: bool = False) -> Any:
        """Deserialize data with optional decompression"""
        if compressed:
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    @staticmethod
    def calculate_checksum(data: bytes) -> str:
        """Calculate checksum for data integrity"""
        return hashlib.sha256(data).hexdigest()

class RedisStateSynchronizer:
    """
    High-performance Redis-based state synchronization system
    
    Features:
    - Real-time synchronization with configurable intervals
    - Conflict resolution using vector clocks
    - Atomic operations with Redis transactions
    - Automatic compression for large states
    - Performance monitoring and alerting
    - Graceful failover handling
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/3",
                 instance_id: str = None,
                 role: InstanceRole = InstanceRole.UNKNOWN,
                 sync_interval: float = 0.5,
                 compression_threshold: int = 1024):
        
        self.redis_url = redis_url
        self.instance_id = instance_id or f"trading_engine_{int(time.time())}"
        self.role = role
        self.sync_interval = sync_interval
        self.fast_sync_enabled = True
        self.critical_sync_interval = 0.1  # 100ms for critical states
        self.compression_threshold = compression_threshold
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        
        # State management
        self.local_state: Dict[StateType, StateSnapshot] = {}
        self.vector_clock = VectorClock()
        self.state_versions: Dict[StateType, int] = defaultdict(int)
        
        # Synchronization
        self.sync_task: Optional[asyncio.Task] = None
        self.pubsub_task: Optional[asyncio.Task] = None
        self.sync_lock = asyncio.Lock()
        self.is_running = False
        
        # Metrics
        self.metrics = SyncMetrics()
        
        # Event bus
        self.event_bus = EventBus()
        
        # Serializer
        self.serializer = StateSerializer()
        
        # Redis keys
        self.state_prefix = "trading_engine:state"
        self.health_key = f"trading_engine:health:{self.instance_id}"
        self.sync_channel = "trading_engine:sync"
        
        logger.info(f"State synchronizer initialized for instance {self.instance_id} with role {self.role.value}")
    
    async def initialize(self):
        """Initialize Redis connections and start synchronization"""
        try:
            # Create Redis clients
            self.redis_client = redis.from_url(self.redis_url)
            self.pubsub_client = redis.from_url(self.redis_url)
            
            # Test connections
            await self.redis_client.ping()
            await self.pubsub_client.ping()
            
            # Register instance
            await self._register_instance()
            
            # Start synchronization
            self.is_running = True
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.pubsub_task = asyncio.create_task(self._pubsub_loop())
            
            logger.info(f"State synchronizer started for instance {self.instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize state synchronizer: {e}")
            raise
    
    async def _register_instance(self):
        """Register this instance in Redis"""
        instance_info = {
            'instance_id': self.instance_id,
            'role': self.role.value,
            'started_at': time.time(),
            'last_heartbeat': time.time()
        }
        
        # Set instance info with TTL
        await self.redis_client.hset(
            self.health_key,
            mapping=instance_info
        )
        await self.redis_client.expire(self.health_key, 60)  # 1 minute TTL
        
        # Publish instance registration
        await self.redis_client.publish(
            self.sync_channel,
            json.dumps({
                'type': 'instance_registered',
                'instance_id': self.instance_id,
                'role': self.role.value,
                'timestamp': time.time()
            })
        )
    
    async def update_state(self, state_type: StateType, data: Dict[str, Any]):
        """Update local state and trigger synchronization"""
        async with self.sync_lock:
            # Update vector clock
            self.vector_clock.tick(self.instance_id)
            
            # Increment version
            self.state_versions[state_type] += 1
            
            # Serialize and compress if needed
            serialized_data = self.serializer.serialize(data, compress=True)
            compressed = len(serialized_data) < len(pickle.dumps(data))
            
            # Calculate checksum
            checksum = self.serializer.calculate_checksum(serialized_data)
            
            # Create state snapshot
            snapshot = StateSnapshot(
                state_type=state_type,
                data=data,
                timestamp=time.time(),
                instance_id=self.instance_id,
                vector_clock=VectorClock(self.vector_clock.clocks.copy()),
                version=self.state_versions[state_type],
                checksum=checksum,
                compressed=compressed
            )
            
            # Store locally
            self.local_state[state_type] = snapshot
            
            # Synchronize immediately for critical states with fast sync
            if state_type in [StateType.TRADING_STATE, StateType.RISK_STATE]:
                await self._sync_state(state_type, snapshot)
                
                # Schedule fast sync for critical states
                if self.fast_sync_enabled:
                    asyncio.create_task(self._fast_sync_critical_states())
            
            logger.debug(f"Updated state {state_type.value} version {snapshot.version}")
    
    async def get_state(self, state_type: StateType) -> Optional[Dict[str, Any]]:
        """Get current state"""
        snapshot = self.local_state.get(state_type)
        return snapshot.data if snapshot else None
    
    async def _sync_state(self, state_type: StateType, snapshot: StateSnapshot):
        """Synchronize state to Redis"""
        sync_start = time.time()
        
        try:
            # Serialize snapshot
            serialized_data = self.serializer.serialize(snapshot.to_dict(), compress=True)
            
            # Redis key
            state_key = f"{self.state_prefix}:{state_type.value}"
            
            # Atomic update using transaction
            async with self.redis_client.pipeline(transaction=True) as pipe:
                # Watch for concurrent updates
                await pipe.watch(state_key)
                
                # Get current state
                current_data = await pipe.get(state_key)
                
                # Check for conflicts
                if current_data:
                    current_snapshot = StateSnapshot.from_dict(
                        self.serializer.deserialize(current_data, compressed=True)
                    )
                    
                    # Resolve conflict using vector clock
                    comparison = snapshot.vector_clock.compare(current_snapshot.vector_clock)
                    
                    if comparison == "less":
                        # Remote state is newer, update local
                        self.local_state[state_type] = current_snapshot
                        self.vector_clock.update(current_snapshot.vector_clock)
                        logger.debug(f"Conflict resolved: remote state newer for {state_type.value}")
                        self.metrics.conflicts_resolved += 1
                        return
                    elif comparison == "concurrent":
                        # Concurrent update, merge states
                        await self._merge_states(state_type, snapshot, current_snapshot)
                        self.metrics.conflicts_resolved += 1
                        return
                
                # Update state
                await pipe.multi()
                await pipe.set(state_key, serialized_data)
                await pipe.expire(state_key, 3600)  # 1 hour TTL
                await pipe.execute()
                
                # Publish change notification
                await self.redis_client.publish(
                    self.sync_channel,
                    json.dumps({
                        'type': 'state_updated',
                        'state_type': state_type.value,
                        'instance_id': self.instance_id,
                        'version': snapshot.version,
                        'timestamp': snapshot.timestamp
                    })
                )
                
                # Update metrics
                sync_time_ms = (time.time() - sync_start) * 1000
                self.metrics.update_sync_time(sync_time_ms)
                self.metrics.successful_syncs += 1
                self.metrics.network_bytes_sent += len(serialized_data)
                
                logger.debug(f"Synchronized state {state_type.value} in {sync_time_ms:.2f}ms")
                
        except redis.exceptions.WatchError:
            # Retry on watch error
            logger.warning(f"Watch error during sync of {state_type.value}, retrying")
            await asyncio.sleep(0.01)  # Small delay
            await self._sync_state(state_type, snapshot)
            
        except Exception as e:
            logger.error(f"Failed to sync state {state_type.value}: {e}")
            self.metrics.failed_syncs += 1
    
    async def _merge_states(self, state_type: StateType, local: StateSnapshot, remote: StateSnapshot):
        """Merge concurrent state updates"""
        logger.info(f"Merging concurrent states for {state_type.value}")
        
        # Simple merge strategy: prefer local for now
        # In production, implement domain-specific merge logic
        merged_data = local.data.copy()
        
        # Update vector clock
        local.vector_clock.update(remote.vector_clock)
        local.vector_clock.tick(self.instance_id)
        
        # Create merged snapshot
        merged_snapshot = StateSnapshot(
            state_type=state_type,
            data=merged_data,
            timestamp=time.time(),
            instance_id=self.instance_id,
            vector_clock=local.vector_clock,
            version=max(local.version, remote.version) + 1,
            checksum=self.serializer.calculate_checksum(pickle.dumps(merged_data))
        )
        
        # Store locally
        self.local_state[state_type] = merged_snapshot
        
        # Sync merged state with priority
        await self._sync_state(state_type, merged_snapshot)
        
        # If it's a critical state, trigger fast sync
        if state_type in [StateType.TRADING_STATE, StateType.RISK_STATE] and self.fast_sync_enabled:
            asyncio.create_task(self._fast_sync_critical_states())
    
    async def _sync_loop(self):
        """Main synchronization loop with adaptive intervals"""
        while self.is_running:
            try:
                # Heartbeat
                await self._send_heartbeat()
                
                # Sync all states with priority-based intervals
                for state_type, snapshot in self.local_state.items():
                    # Use faster sync for critical states
                    if state_type in [StateType.TRADING_STATE, StateType.RISK_STATE]:
                        await self._sync_state(state_type, snapshot)
                    else:
                        # Less frequent sync for non-critical states
                        if int(time.time() * 2) % 2 == 0:  # Every other cycle
                            await self._sync_state(state_type, snapshot)
                
                # Adaptive sleep based on system load
                sync_interval = self.sync_interval
                if self.metrics.failed_syncs > 0 and self.metrics.total_syncs > 0:
                    failure_rate = self.metrics.failed_syncs / self.metrics.total_syncs
                    if failure_rate > 0.1:  # If >10% failure rate, slow down
                        sync_interval *= 1.5
                    elif failure_rate < 0.01:  # If <1% failure rate, speed up
                        sync_interval *= 0.8
                
                await asyncio.sleep(min(sync_interval, 2.0))  # Cap at 2 seconds
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(0.5)  # Shorter pause on error
    
    async def _send_heartbeat(self):
        """Send heartbeat to indicate instance is alive"""
        try:
            await self.redis_client.hset(
                self.health_key,
                'last_heartbeat',
                time.time()
            )
            await self.redis_client.expire(self.health_key, 60)
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def _pubsub_loop(self):
        """Listen for state change notifications"""
        try:
            pubsub = self.pubsub_client.pubsub()
            await pubsub.subscribe(self.sync_channel)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self._handle_sync_message(data)
                        
                    except Exception as e:
                        logger.error(f"Error handling sync message: {e}")
                        
        except Exception as e:
            logger.error(f"Error in pubsub loop: {e}")
    
    async def _handle_sync_message(self, message: Dict[str, Any]):
        """Handle synchronization message"""
        message_type = message.get('type')
        
        if message_type == 'state_updated':
            state_type = StateType(message['state_type'])
            instance_id = message['instance_id']
            
            # Skip our own messages
            if instance_id == self.instance_id:
                return
            
            # Fetch updated state
            await self._fetch_remote_state(state_type)
            
        elif message_type == 'instance_registered':
            instance_id = message['instance_id']
            role = message['role']
            logger.info(f"New instance registered: {instance_id} ({role})")
            
            # Emit event
            await self.event_bus.emit(Event(
                type=EventType.SYSTEM_READY,
                data={'instance_id': instance_id, 'role': role}
            ))
    
    async def _fetch_remote_state(self, state_type: StateType):
        """Fetch state from Redis"""
        try:
            state_key = f"{self.state_prefix}:{state_type.value}"
            remote_data = await self.redis_client.get(state_key)
            
            if remote_data:
                remote_snapshot = StateSnapshot.from_dict(
                    self.serializer.deserialize(remote_data, compressed=True)
                )
                
                # Check if remote is newer
                local_snapshot = self.local_state.get(state_type)
                if not local_snapshot or remote_snapshot.version > local_snapshot.version:
                    self.local_state[state_type] = remote_snapshot
                    self.vector_clock.update(remote_snapshot.vector_clock)
                    
                    logger.debug(f"Updated local state {state_type.value} from remote")
                    
                    # Emit state change event
                    await self.event_bus.emit(Event(
                        type=EventType.STATE_CHANGED,
                        data={
                            'state_type': state_type.value,
                            'version': remote_snapshot.version,
                            'instance_id': remote_snapshot.instance_id
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to fetch remote state {state_type.value}: {e}")
    
    async def get_all_instances(self) -> List[Dict[str, Any]]:
        """Get all registered instances"""
        try:
            keys = await self.redis_client.keys("trading_engine:health:*")
            instances = []
            
            for key in keys:
                instance_data = await self.redis_client.hgetall(key)
                if instance_data:
                    instance_info = {
                        'instance_id': instance_data.get('instance_id'),
                        'role': instance_data.get('role'),
                        'started_at': float(instance_data.get('started_at', 0)),
                        'last_heartbeat': float(instance_data.get('last_heartbeat', 0))
                    }
                    instances.append(instance_info)
            
            return instances
            
        except Exception as e:
            logger.error(f"Failed to get instances: {e}")
            return []
    
    async def promote_to_active(self):
        """Promote this instance to active role"""
        if self.role == InstanceRole.ACTIVE:
            return
        
        logger.info(f"Promoting instance {self.instance_id} to active role")
        
        self.role = InstanceRole.ACTIVE
        
        # Update instance info
        await self.redis_client.hset(
            self.health_key,
            'role',
            self.role.value
        )
        
        # Publish promotion event
        await self.redis_client.publish(
            self.sync_channel,
            json.dumps({
                'type': 'instance_promoted',
                'instance_id': self.instance_id,
                'role': self.role.value,
                'timestamp': time.time()
            })
        )
        
        # Emit event
        await self.event_bus.emit(Event(
            type=EventType.FAILOVER_COMPLETED,
            data={'instance_id': self.instance_id, 'new_role': self.role.value}
        ))
    
    async def demote_to_passive(self):
        """Demote this instance to passive role"""
        if self.role == InstanceRole.PASSIVE:
            return
        
        logger.info(f"Demoting instance {self.instance_id} to passive role")
        
        self.role = InstanceRole.PASSIVE
        
        # Update instance info
        await self.redis_client.hset(
            self.health_key,
            'role',
            self.role.value
        )
        
        # Publish demotion event
        await self.redis_client.publish(
            self.sync_channel,
            json.dumps({
                'type': 'instance_demoted',
                'instance_id': self.instance_id,
                'role': self.role.value,
                'timestamp': time.time()
            })
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get synchronization metrics"""
        return {
            'total_syncs': self.metrics.total_syncs,
            'successful_syncs': self.metrics.successful_syncs,
            'failed_syncs': self.metrics.failed_syncs,
            'success_rate': self.metrics.successful_syncs / max(1, self.metrics.total_syncs),
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'avg_sync_time_ms': self.metrics.avg_sync_time_ms,
            'max_sync_time_ms': self.metrics.max_sync_time_ms,
            'network_bytes_sent': self.metrics.network_bytes_sent,
            'network_bytes_received': self.metrics.network_bytes_received,
            'last_sync_time': self.metrics.last_sync_time,
            'instance_id': self.instance_id,
            'role': self.role.value,
            'states_count': len(self.local_state)
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down state synchronizer for instance {self.instance_id}")
        
        self.is_running = False
        
        # Cancel tasks
        if self.sync_task:
            self.sync_task.cancel()
        if self.pubsub_task:
            self.pubsub_task.cancel()
        
        # Remove instance from Redis
        try:
            await self.redis_client.delete(self.health_key)
        except:
            pass
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
        if self.pubsub_client:
            await self.pubsub_client.close()
        
        logger.info("State synchronizer shutdown complete")
    
    async def _fast_sync_critical_states(self):
        """Fast synchronization for critical states"""
        try:
            # Fast sync only critical states
            critical_states = [
                StateType.TRADING_STATE,
                StateType.RISK_STATE,
                StateType.CIRCUIT_BREAKER_STATE
            ]
            
            for state_type in critical_states:
                if state_type in self.local_state:
                    snapshot = self.local_state[state_type]
                    await self._sync_state(state_type, snapshot)
                    
        except Exception as e:
            logger.error(f"Error in fast sync: {e}")
    
    async def enable_fast_sync(self):
        """Enable fast synchronization mode"""
        self.fast_sync_enabled = True
        self.sync_interval = min(self.sync_interval, 0.25)  # Max 250ms
        logger.info("Fast sync mode enabled")
    
    async def disable_fast_sync(self):
        """Disable fast synchronization mode"""
        self.fast_sync_enabled = False
        logger.info("Fast sync mode disabled")
    
    async def force_full_sync(self):
        """Force immediate full synchronization"""
        logger.info("Forcing full synchronization")
        for state_type, snapshot in self.local_state.items():
            await self._sync_state(state_type, snapshot)
    
    async def get_sync_health(self) -> Dict[str, Any]:
        """Get synchronization health metrics"""
        current_time = time.time()
        
        # Calculate sync latency
        sync_latency = current_time - self.metrics.last_sync_time if self.metrics.last_sync_time > 0 else 0
        
        # Calculate health score
        health_score = 100.0
        
        # Penalize high sync latency
        if sync_latency > 1.0:
            health_score -= min(50, sync_latency * 20)
        
        # Penalize high failure rate
        if self.metrics.total_syncs > 0:
            failure_rate = self.metrics.failed_syncs / self.metrics.total_syncs
            health_score -= failure_rate * 100
        
        # Penalize slow sync times
        if self.metrics.avg_sync_time_ms > 100:
            health_score -= min(30, (self.metrics.avg_sync_time_ms - 100) / 10)
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'sync_latency': sync_latency,
            'failure_rate': self.metrics.failed_syncs / max(1, self.metrics.total_syncs),
            'avg_sync_time_ms': self.metrics.avg_sync_time_ms,
            'fast_sync_enabled': self.fast_sync_enabled,
            'states_count': len(self.local_state)
        }


# Factory function
def create_state_synchronizer(
    redis_url: str,
    instance_id: str,
    role: InstanceRole,
    sync_interval: float = 0.5
) -> RedisStateSynchronizer:
    """Create Redis state synchronizer"""
    return RedisStateSynchronizer(
        redis_url=redis_url,
        instance_id=instance_id,
        role=role,
        sync_interval=sync_interval
    )


# CLI for testing
async def main():
    """Test the state synchronizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Redis State Synchronizer")
    parser.add_argument("--redis-url", default="redis://localhost:6379/3")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--role", choices=["active", "passive"], default="passive")
    parser.add_argument("--sync-interval", type=float, default=0.5)
    
    args = parser.parse_args()
    
    role = InstanceRole.ACTIVE if args.role == "active" else InstanceRole.PASSIVE
    
    # Create synchronizer
    sync = create_state_synchronizer(
        redis_url=args.redis_url,
        instance_id=args.instance_id,
        role=role,
        sync_interval=args.sync_interval
    )
    
    # Initialize
    await sync.initialize()
    
    # Test state updates
    test_data = {
        'portfolio_value': 100000.0,
        'positions': {'AAPL': 100, 'GOOGL': 50},
        'timestamp': time.time()
    }
    
    await sync.update_state(StateType.TRADING_STATE, test_data)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(5)
            
            # Update test data
            test_data['timestamp'] = time.time()
            test_data['portfolio_value'] += 100
            
            await sync.update_state(StateType.TRADING_STATE, test_data)
            
            # Print metrics
            metrics = sync.get_metrics()
            print(f"Metrics: {metrics}")
            
    except KeyboardInterrupt:
        await sync.shutdown()


if __name__ == "__main__":
    asyncio.run(main())