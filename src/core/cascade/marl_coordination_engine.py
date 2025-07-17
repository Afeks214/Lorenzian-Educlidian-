"""
MARL Coordination Engine - Inter-system coordination and communication

This module provides sophisticated coordination between MARL systems,
ensuring seamless communication and synchronization across the cascade.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from datetime import datetime, timedelta
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import json
import hashlib
import uuid

from ..events import EventBus, Event, EventType
from ..errors import BaseException as CoreBaseException
from ..performance.performance_monitor import PerformanceMonitor
from .superposition_cascade_manager import SuperpositionPacket, SuperpositionType


class CoordinationState(Enum):
    """Coordination states between MARL systems"""
    INITIALIZING = "INITIALIZING"
    SYNCHRONIZED = "SYNCHRONIZED"
    COORDINATING = "COORDINATING"
    CONFLICT = "CONFLICT"
    EMERGENCY = "EMERGENCY"
    DISCONNECTED = "DISCONNECTED"


class MessageType(Enum):
    """Inter-MARL communication message types"""
    HANDSHAKE = "HANDSHAKE"
    HEARTBEAT = "HEARTBEAT"
    SYNC_REQUEST = "SYNC_REQUEST"
    SYNC_RESPONSE = "SYNC_RESPONSE"
    COORDINATION_REQUEST = "COORDINATION_REQUEST"
    COORDINATION_RESPONSE = "COORDINATION_RESPONSE"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    EMERGENCY_BROADCAST = "EMERGENCY_BROADCAST"
    STATE_UPDATE = "STATE_UPDATE"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"


@dataclass
class CoordinationMessage:
    """Standard inter-MARL coordination message"""
    message_id: str
    message_type: MessageType
    sender_system: str
    recipient_system: str
    timestamp: datetime
    payload: Dict[str, Any]
    priority: int = 1
    requires_response: bool = False
    response_timeout: float = 5.0
    correlation_id: Optional[str] = None
    

@dataclass
class SystemState:
    """State information for a MARL system"""
    system_id: str
    system_name: str
    state: CoordinationState
    last_heartbeat: datetime
    performance_metrics: Dict[str, float]
    capabilities: List[str]
    current_load: float
    version: str
    configuration: Dict[str, Any]


@dataclass
class CoordinationMetrics:
    """Coordination engine metrics"""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    conflict_resolutions: int = 0
    average_response_time_ms: float = 0.0
    system_sync_score: float = 100.0
    coordination_efficiency: float = 100.0


class MARLCoordinationEngine:
    """
    Advanced coordination engine for managing communication and synchronization
    between Strategic, Tactical, Risk, and Execution MARL systems.
    """

    def __init__(
        self,
        event_bus: EventBus,
        performance_monitor: Optional[PerformanceMonitor] = None,
        heartbeat_interval: float = 1.0,
        sync_timeout: float = 10.0,
        max_concurrent_requests: int = 50
    ):
        self.event_bus = event_bus
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.heartbeat_interval = heartbeat_interval
        self.sync_timeout = sync_timeout
        self.max_concurrent_requests = max_concurrent_requests
        
        # State management
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # System registry
        self._systems: Dict[str, SystemState] = {}
        self._coordination_state = CoordinationState.INITIALIZING
        
        # Communication infrastructure
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._response_waiters: Dict[str, asyncio.Future] = {}
        self._pending_requests: Dict[str, CoordinationMessage] = {}
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # Metrics
        self._metrics = CoordinationMetrics()
        self._response_times: List[float] = []
        
        # Coordination protocols
        self._conflict_resolution_strategies: Dict[str, Callable] = {}
        self._sync_protocols: Dict[str, Callable] = {}
        
        # Initialize engine
        self._initialize_coordination_engine()
        
    def _initialize_coordination_engine(self) -> None:
        """Initialize the coordination engine"""
        try:
            # Register message handlers
            self._register_message_handlers()
            
            # Initialize conflict resolution strategies
            self._initialize_conflict_resolution()
            
            # Initialize sync protocols
            self._initialize_sync_protocols()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Register for system events
            self._register_event_handlers()
            
            self._coordination_state = CoordinationState.SYNCHRONIZED
            self.logger.info("MARL coordination engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordination engine: {e}")
            self._coordination_state = CoordinationState.EMERGENCY
            raise
            
    def _register_message_handlers(self) -> None:
        """Register handlers for different message types"""
        self._message_handlers = {
            MessageType.HANDSHAKE: self._handle_handshake,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.SYNC_REQUEST: self._handle_sync_request,
            MessageType.SYNC_RESPONSE: self._handle_sync_response,
            MessageType.COORDINATION_REQUEST: self._handle_coordination_request,
            MessageType.COORDINATION_RESPONSE: self._handle_coordination_response,
            MessageType.CONFLICT_RESOLUTION: self._handle_conflict_resolution,
            MessageType.EMERGENCY_BROADCAST: self._handle_emergency_broadcast,
            MessageType.STATE_UPDATE: self._handle_state_update,
            MessageType.PERFORMANCE_REPORT: self._handle_performance_report
        }
        
    def _initialize_conflict_resolution(self) -> None:
        """Initialize conflict resolution strategies"""
        self._conflict_resolution_strategies = {
            "priority_based": self._priority_based_resolution,
            "performance_based": self._performance_based_resolution,
            "consensus_based": self._consensus_based_resolution,
            "hierarchical": self._hierarchical_resolution
        }
        
    def _initialize_sync_protocols(self) -> None:
        """Initialize synchronization protocols"""
        self._sync_protocols = {
            "fast_sync": self._fast_sync_protocol,
            "reliable_sync": self._reliable_sync_protocol,
            "emergency_sync": self._emergency_sync_protocol
        }
        
    def _start_background_tasks(self) -> None:
        """Start background coordination tasks"""
        threading.Thread(target=self._heartbeat_sender, daemon=True).start()
        threading.Thread(target=self._system_monitor, daemon=True).start()
        threading.Thread(target=self._coordination_monitor, daemon=True).start()
        threading.Thread(target=self._metrics_updater, daemon=True).start()
        
    def _register_event_handlers(self) -> None:
        """Register event handlers for system events"""
        self.event_bus.subscribe(EventType.SYSTEM_START, self._handle_system_start)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._handle_system_shutdown)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
        
    def register_marl_system(
        self,
        system_id: str,
        system_name: str,
        capabilities: List[str],
        configuration: Dict[str, Any],
        version: str = "1.0.0"
    ) -> None:
        """Register a MARL system with the coordination engine"""
        with self._lock:
            system_state = SystemState(
                system_id=system_id,
                system_name=system_name,
                state=CoordinationState.INITIALIZING,
                last_heartbeat=datetime.now(),
                performance_metrics={},
                capabilities=capabilities,
                current_load=0.0,
                version=version,
                configuration=configuration
            )
            
            self._systems[system_id] = system_state
            
            # Send handshake to system
            self._send_handshake(system_id)
            
            self.logger.info(f"MARL system registered: {system_name} ({system_id})")
            
    def _send_handshake(self, system_id: str) -> None:
        """Send handshake message to a system"""
        message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HANDSHAKE,
            sender_system="coordination_engine",
            recipient_system=system_id,
            timestamp=datetime.now(),
            payload={
                "engine_version": "1.0.0",
                "supported_protocols": list(self._sync_protocols.keys()),
                "coordination_capabilities": list(self._conflict_resolution_strategies.keys())
            },
            requires_response=True
        )
        
        self._send_message(message)
        
    def coordinate_decision(
        self,
        requesting_system: str,
        decision_type: str,
        decision_data: Dict[str, Any],
        affected_systems: List[str],
        priority: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Coordinate a decision across multiple MARL systems"""
        coordination_id = str(uuid.uuid4())
        
        try:
            # Create coordination request
            coordination_request = {
                "coordination_id": coordination_id,
                "requesting_system": requesting_system,
                "decision_type": decision_type,
                "decision_data": decision_data,
                "affected_systems": affected_systems,
                "timestamp": datetime.now().isoformat(),
                "priority": priority
            }
            
            # Send coordination requests to affected systems
            responses = self._send_coordination_requests(coordination_request)
            
            # Process responses and resolve conflicts
            result = self._process_coordination_responses(coordination_request, responses)
            
            # Update metrics
            if result.get("success", False):
                self._metrics.successful_coordinations += 1
            else:
                self._metrics.failed_coordinations += 1
                
            return result
            
        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            self._metrics.failed_coordinations += 1
            return {"success": False, "error": str(e)}
            
    def _send_coordination_requests(
        self, 
        coordination_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send coordination requests to affected systems"""
        responses = {}
        futures = []
        
        for system_id in coordination_request["affected_systems"]:
            if system_id in self._systems:
                message = CoordinationMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.COORDINATION_REQUEST,
                    sender_system="coordination_engine",
                    recipient_system=system_id,
                    timestamp=datetime.now(),
                    payload=coordination_request,
                    requires_response=True,
                    correlation_id=coordination_request["coordination_id"]
                )
                
                future = self._executor.submit(self._send_message_with_response, message)
                futures.append((system_id, future))
                
        # Collect responses
        for system_id, future in futures:
            try:
                response = future.result(timeout=self.sync_timeout)
                responses[system_id] = response
            except Exception as e:
                self.logger.error(f"Failed to get response from {system_id}: {e}")
                responses[system_id] = {"success": False, "error": str(e)}
                
        return responses
        
    def _process_coordination_responses(
        self,
        coordination_request: Dict[str, Any],
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process coordination responses and resolve conflicts"""
        # Check for conflicts
        conflicts = self._detect_conflicts(responses)
        
        if conflicts:
            # Resolve conflicts using appropriate strategy
            resolution_strategy = self._select_resolution_strategy(coordination_request, conflicts)
            resolved_decision = self._resolve_conflicts(conflicts, resolution_strategy)
            
            if resolved_decision:
                # Broadcast resolution to all systems
                self._broadcast_resolution(coordination_request["coordination_id"], resolved_decision)
                return {"success": True, "decision": resolved_decision, "conflicts_resolved": len(conflicts)}
            else:
                return {"success": False, "error": "Unable to resolve conflicts"}
        else:
            # No conflicts, aggregate responses
            aggregated_decision = self._aggregate_responses(responses)
            return {"success": True, "decision": aggregated_decision}
            
    def _detect_conflicts(self, responses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts in coordination responses"""
        conflicts = []
        decisions = []
        
        for system_id, response in responses.items():
            if response.get("success", False) and "decision" in response:
                decision = response["decision"]
                decision["system_id"] = system_id
                decisions.append(decision)
                
        # Compare decisions for conflicts
        for i, decision1 in enumerate(decisions):
            for j, decision2 in enumerate(decisions[i+1:], i+1):
                if self._are_decisions_conflicting(decision1, decision2):
                    conflicts.append({
                        "systems": [decision1["system_id"], decision2["system_id"]],
                        "decisions": [decision1, decision2],
                        "conflict_type": self._classify_conflict(decision1, decision2)
                    })
                    
        return conflicts
        
    def _are_decisions_conflicting(self, decision1: Dict[str, Any], decision2: Dict[str, Any]) -> bool:
        """Check if two decisions are conflicting"""
        # Check for resource conflicts
        if "resources" in decision1 and "resources" in decision2:
            resources1 = set(decision1["resources"])
            resources2 = set(decision2["resources"])
            if resources1 & resources2:  # Intersection means conflict
                return True
                
        # Check for parameter conflicts
        if "parameters" in decision1 and "parameters" in decision2:
            params1 = decision1["parameters"]
            params2 = decision2["parameters"]
            
            for key in params1:
                if key in params2 and params1[key] != params2[key]:
                    return True
                    
        # Check for timing conflicts
        if "timing" in decision1 and "timing" in decision2:
            timing1 = decision1["timing"]
            timing2 = decision2["timing"]
            
            # Check for overlapping time windows
            if self._time_windows_overlap(timing1, timing2):
                return True
                
        return False
        
    def _classify_conflict(self, decision1: Dict[str, Any], decision2: Dict[str, Any]) -> str:
        """Classify the type of conflict"""
        if "resources" in decision1 and "resources" in decision2:
            return "resource_conflict"
        elif "parameters" in decision1 and "parameters" in decision2:
            return "parameter_conflict"
        elif "timing" in decision1 and "timing" in decision2:
            return "timing_conflict"
        else:
            return "general_conflict"
            
    def _time_windows_overlap(self, timing1: Dict[str, Any], timing2: Dict[str, Any]) -> bool:
        """Check if two time windows overlap"""
        try:
            start1 = datetime.fromisoformat(timing1.get("start", "1970-01-01T00:00:00"))
            end1 = datetime.fromisoformat(timing1.get("end", "2100-01-01T00:00:00"))
            start2 = datetime.fromisoformat(timing2.get("start", "1970-01-01T00:00:00"))
            end2 = datetime.fromisoformat(timing2.get("end", "2100-01-01T00:00:00"))
            
            return start1 < end2 and start2 < end1
        except Exception:
            return False
            
    def _select_resolution_strategy(
        self, 
        coordination_request: Dict[str, Any], 
        conflicts: List[Dict[str, Any]]
    ) -> str:
        """Select appropriate conflict resolution strategy"""
        # Priority-based resolution for high-priority decisions
        if coordination_request.get("priority", 1) >= 2:
            return "priority_based"
            
        # Performance-based resolution for performance-critical decisions
        if coordination_request.get("decision_type") in ["execution", "risk"]:
            return "performance_based"
            
        # Consensus-based resolution for strategic decisions
        if coordination_request.get("decision_type") == "strategic":
            return "consensus_based"
            
        # Default to hierarchical resolution
        return "hierarchical"
        
    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]], strategy: str) -> Optional[Dict[str, Any]]:
        """Resolve conflicts using specified strategy"""
        resolver = self._conflict_resolution_strategies.get(strategy)
        if resolver:
            return resolver(conflicts)
        else:
            self.logger.error(f"Unknown resolution strategy: {strategy}")
            return None
            
    def _priority_based_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts based on system priorities"""
        system_priorities = {
            "strategic": 1,
            "tactical": 2,
            "risk": 3,
            "execution": 4
        }
        
        best_decision = None
        best_priority = float('inf')
        
        for conflict in conflicts:
            for decision in conflict["decisions"]:
                system_id = decision["system_id"]
                priority = system_priorities.get(system_id, 999)
                
                if priority < best_priority:
                    best_priority = priority
                    best_decision = decision
                    
        return best_decision
        
    def _performance_based_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts based on system performance"""
        best_decision = None
        best_performance = 0
        
        for conflict in conflicts:
            for decision in conflict["decisions"]:
                system_id = decision["system_id"]
                system_state = self._systems.get(system_id)
                
                if system_state:
                    # Calculate performance score
                    performance_score = self._calculate_performance_score(system_state)
                    
                    if performance_score > best_performance:
                        best_performance = performance_score
                        best_decision = decision
                        
        return best_decision
        
    def _consensus_based_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts based on consensus"""
        decision_votes = {}
        
        for conflict in conflicts:
            for decision in conflict["decisions"]:
                decision_hash = self._hash_decision(decision)
                if decision_hash not in decision_votes:
                    decision_votes[decision_hash] = {"decision": decision, "votes": 0}
                decision_votes[decision_hash]["votes"] += 1
                
        # Return decision with most votes
        if decision_votes:
            best_decision = max(decision_votes.values(), key=lambda x: x["votes"])
            return best_decision["decision"]
            
        return None
        
    def _hierarchical_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts using hierarchical approach"""
        # Use system hierarchy: strategic > tactical > risk > execution
        hierarchy = ["strategic", "tactical", "risk", "execution"]
        
        for system_id in hierarchy:
            for conflict in conflicts:
                for decision in conflict["decisions"]:
                    if decision["system_id"] == system_id:
                        return decision
                        
        return None
        
    def _calculate_performance_score(self, system_state: SystemState) -> float:
        """Calculate performance score for a system"""
        metrics = system_state.performance_metrics
        
        # Weighted performance score
        score = 0.0
        weights = {
            "response_time": -0.3,  # Lower is better
            "throughput": 0.3,      # Higher is better
            "accuracy": 0.2,        # Higher is better
            "availability": 0.2     # Higher is better
        }
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
                
        return score
        
    def _hash_decision(self, decision: Dict[str, Any]) -> str:
        """Create hash of decision for comparison"""
        # Remove system_id for hashing
        decision_copy = decision.copy()
        decision_copy.pop("system_id", None)
        
        decision_str = json.dumps(decision_copy, sort_keys=True)
        return hashlib.md5(decision_str.encode()).hexdigest()
        
    def _aggregate_responses(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses when no conflicts exist"""
        aggregated = {}
        
        for system_id, response in responses.items():
            if response.get("success", False) and "decision" in response:
                decision = response["decision"]
                for key, value in decision.items():
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].append(value)
                    
        # Process aggregated data
        final_decision = {}
        for key, values in aggregated.items():
            if len(values) == 1:
                final_decision[key] = values[0]
            else:
                # Use appropriate aggregation method
                if all(isinstance(v, (int, float)) for v in values):
                    final_decision[key] = sum(values) / len(values)  # Average
                else:
                    final_decision[key] = values[0]  # First value
                    
        return final_decision
        
    def _broadcast_resolution(self, coordination_id: str, resolution: Dict[str, Any]) -> None:
        """Broadcast conflict resolution to all systems"""
        for system_id in self._systems:
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.CONFLICT_RESOLUTION,
                sender_system="coordination_engine",
                recipient_system=system_id,
                timestamp=datetime.now(),
                payload={
                    "coordination_id": coordination_id,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat()
                },
                correlation_id=coordination_id
            )
            
            self._send_message(message)
            
    def _send_message(self, message: CoordinationMessage) -> None:
        """Send coordination message to target system"""
        try:
            # Log message
            self.logger.debug(f"Sending {message.message_type.value} to {message.recipient_system}")
            
            # Handle message based on type
            handler = self._message_handlers.get(message.message_type)
            if handler:
                handler(message)
                
            # Update metrics
            self._metrics.total_messages_sent += 1
            
            # Publish event for monitoring
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.SYSTEM_START,  # Use appropriate event type
                    {
                        "type": "coordination_message",
                        "message_type": message.message_type.value,
                        "sender": message.sender_system,
                        "recipient": message.recipient_system
                    },
                    "coordination_engine"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            
    def _send_message_with_response(self, message: CoordinationMessage) -> Optional[Dict[str, Any]]:
        """Send message and wait for response"""
        if not message.requires_response:
            self._send_message(message)
            return None
            
        # Create response waiter
        response_future = asyncio.Future()
        self._response_waiters[message.message_id] = response_future
        
        # Send message
        self._send_message(message)
        
        # Wait for response
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                asyncio.wait_for(response_future, timeout=message.response_timeout)
            )
            return response
        except asyncio.TimeoutError:
            self.logger.error(f"Response timeout for message {message.message_id}")
            return None
        finally:
            self._response_waiters.pop(message.message_id, None)
            
    def synchronize_systems(self, sync_protocol: str = "fast_sync") -> Dict[str, bool]:
        """Synchronize all MARL systems"""
        sync_handler = self._sync_protocols.get(sync_protocol)
        if not sync_handler:
            self.logger.error(f"Unknown sync protocol: {sync_protocol}")
            return {}
            
        return sync_handler()
        
    def _fast_sync_protocol(self) -> Dict[str, bool]:
        """Fast synchronization protocol"""
        results = {}
        
        for system_id in self._systems:
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SYNC_REQUEST,
                sender_system="coordination_engine",
                recipient_system=system_id,
                timestamp=datetime.now(),
                payload={"sync_type": "fast", "timestamp": datetime.now().isoformat()},
                requires_response=True,
                response_timeout=2.0
            )
            
            response = self._send_message_with_response(message)
            results[system_id] = response is not None and response.get("success", False)
            
        return results
        
    def _reliable_sync_protocol(self) -> Dict[str, bool]:
        """Reliable synchronization protocol"""
        results = {}
        
        for system_id in self._systems:
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SYNC_REQUEST,
                sender_system="coordination_engine",
                recipient_system=system_id,
                timestamp=datetime.now(),
                payload={"sync_type": "reliable", "timestamp": datetime.now().isoformat()},
                requires_response=True,
                response_timeout=10.0
            )
            
            response = self._send_message_with_response(message)
            results[system_id] = response is not None and response.get("success", False)
            
        return results
        
    def _emergency_sync_protocol(self) -> Dict[str, bool]:
        """Emergency synchronization protocol"""
        results = {}
        
        # Broadcast emergency sync to all systems
        for system_id in self._systems:
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.EMERGENCY_BROADCAST,
                sender_system="coordination_engine",
                recipient_system=system_id,
                timestamp=datetime.now(),
                payload={
                    "emergency_type": "sync_required",
                    "timestamp": datetime.now().isoformat(),
                    "priority": 3
                },
                priority=3
            )
            
            self._send_message(message)
            results[system_id] = True  # Assume success for emergency broadcast
            
        return results
        
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "coordination_state": self._coordination_state.value,
            "registered_systems": len(self._systems),
            "active_systems": len([s for s in self._systems.values() if s.state == CoordinationState.SYNCHRONIZED]),
            "metrics": {
                "total_messages_sent": self._metrics.total_messages_sent,
                "total_messages_received": self._metrics.total_messages_received,
                "successful_coordinations": self._metrics.successful_coordinations,
                "failed_coordinations": self._metrics.failed_coordinations,
                "conflict_resolutions": self._metrics.conflict_resolutions,
                "average_response_time_ms": self._metrics.average_response_time_ms,
                "system_sync_score": self._metrics.system_sync_score,
                "coordination_efficiency": self._metrics.coordination_efficiency
            },
            "systems": {
                system_id: {
                    "name": system.system_name,
                    "state": system.state.value,
                    "last_heartbeat": system.last_heartbeat.isoformat(),
                    "current_load": system.current_load,
                    "version": system.version,
                    "capabilities": system.capabilities
                }
                for system_id, system in self._systems.items()
            }
        }
        
    def get_metrics(self) -> CoordinationMetrics:
        """Get coordination metrics"""
        return self._metrics
        
    # Background tasks
    def _heartbeat_sender(self) -> None:
        """Send heartbeat messages to all systems"""
        while not self._shutdown_event.is_set():
            try:
                for system_id in self._systems:
                    message = CoordinationMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.HEARTBEAT,
                        sender_system="coordination_engine",
                        recipient_system=system_id,
                        timestamp=datetime.now(),
                        payload={"timestamp": datetime.now().isoformat()}
                    )
                    
                    self._send_message(message)
                    
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat sender: {e}")
                
    def _system_monitor(self) -> None:
        """Monitor system health and connectivity"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                for system_id, system in self._systems.items():
                    # Check for stale heartbeats
                    time_since_heartbeat = (current_time - system.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_interval * 3:
                        if system.state != CoordinationState.DISCONNECTED:
                            system.state = CoordinationState.DISCONNECTED
                            self.logger.warning(f"System {system_id} disconnected")
                            
                    elif system.state == CoordinationState.DISCONNECTED:
                        system.state = CoordinationState.SYNCHRONIZED
                        self.logger.info(f"System {system_id} reconnected")
                        
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                
    def _coordination_monitor(self) -> None:
        """Monitor coordination performance"""
        while not self._shutdown_event.is_set():
            try:
                # Calculate system sync score
                active_systems = sum(1 for s in self._systems.values() if s.state == CoordinationState.SYNCHRONIZED)
                total_systems = len(self._systems)
                
                if total_systems > 0:
                    self._metrics.system_sync_score = (active_systems / total_systems) * 100
                    
                # Calculate coordination efficiency
                total_coordinations = self._metrics.successful_coordinations + self._metrics.failed_coordinations
                if total_coordinations > 0:
                    self._metrics.coordination_efficiency = (self._metrics.successful_coordinations / total_coordinations) * 100
                    
                time.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in coordination monitor: {e}")
                
    def _metrics_updater(self) -> None:
        """Update coordination metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Update average response time
                if self._response_times:
                    self._metrics.average_response_time_ms = sum(self._response_times) / len(self._response_times) * 1000
                    
                    # Keep only recent response times
                    if len(self._response_times) > 1000:
                        self._response_times = self._response_times[-1000:]
                        
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in metrics updater: {e}")
                
    # Message handlers
    def _handle_handshake(self, message: CoordinationMessage) -> None:
        """Handle handshake message"""
        system_id = message.sender_system
        
        if system_id in self._systems:
            system = self._systems[system_id]
            system.state = CoordinationState.SYNCHRONIZED
            system.last_heartbeat = datetime.now()
            
            self.logger.info(f"Handshake completed with {system_id}")
            
    def _handle_heartbeat(self, message: CoordinationMessage) -> None:
        """Handle heartbeat message"""
        system_id = message.sender_system
        
        if system_id in self._systems:
            self._systems[system_id].last_heartbeat = datetime.now()
            
    def _handle_sync_request(self, message: CoordinationMessage) -> None:
        """Handle sync request message"""
        # Process sync request and send response
        response = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SYNC_RESPONSE,
            sender_system="coordination_engine",
            recipient_system=message.sender_system,
            timestamp=datetime.now(),
            payload={"success": True, "timestamp": datetime.now().isoformat()},
            correlation_id=message.message_id
        )
        
        self._send_message(response)
        
    def _handle_sync_response(self, message: CoordinationMessage) -> None:
        """Handle sync response message"""
        # Process sync response
        if message.correlation_id in self._response_waiters:
            future = self._response_waiters[message.correlation_id]
            if not future.done():
                future.set_result(message.payload)
                
    def _handle_coordination_request(self, message: CoordinationMessage) -> None:
        """Handle coordination request message"""
        # Process coordination request
        response_payload = {
            "success": True,
            "decision": message.payload,
            "timestamp": datetime.now().isoformat()
        }
        
        response = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COORDINATION_RESPONSE,
            sender_system="coordination_engine",
            recipient_system=message.sender_system,
            timestamp=datetime.now(),
            payload=response_payload,
            correlation_id=message.message_id
        )
        
        self._send_message(response)
        
    def _handle_coordination_response(self, message: CoordinationMessage) -> None:
        """Handle coordination response message"""
        if message.correlation_id in self._response_waiters:
            future = self._response_waiters[message.correlation_id]
            if not future.done():
                future.set_result(message.payload)
                
    def _handle_conflict_resolution(self, message: CoordinationMessage) -> None:
        """Handle conflict resolution message"""
        self._metrics.conflict_resolutions += 1
        self.logger.info(f"Conflict resolution processed: {message.payload.get('coordination_id')}")
        
    def _handle_emergency_broadcast(self, message: CoordinationMessage) -> None:
        """Handle emergency broadcast message"""
        self.logger.critical(f"Emergency broadcast received: {message.payload}")
        
        # Update coordination state
        self._coordination_state = CoordinationState.EMERGENCY
        
    def _handle_state_update(self, message: CoordinationMessage) -> None:
        """Handle state update message"""
        system_id = message.sender_system
        
        if system_id in self._systems:
            system = self._systems[system_id]
            system.performance_metrics.update(message.payload.get("metrics", {}))
            system.current_load = message.payload.get("load", 0.0)
            system.last_heartbeat = datetime.now()
            
    def _handle_performance_report(self, message: CoordinationMessage) -> None:
        """Handle performance report message"""
        system_id = message.sender_system
        
        if system_id in self._systems:
            system = self._systems[system_id]
            system.performance_metrics.update(message.payload)
            
    # Event handlers
    def _handle_system_start(self, event: Event) -> None:
        """Handle system start event"""
        self.logger.info("System start event received")
        
    def _handle_system_shutdown(self, event: Event) -> None:
        """Handle system shutdown event"""
        self.logger.info("System shutdown event received")
        self.shutdown()
        
    def _handle_emergency_stop(self, event: Event) -> None:
        """Handle emergency stop event"""
        self.logger.critical("Emergency stop event received")
        self._coordination_state = CoordinationState.EMERGENCY
        
        # Broadcast emergency to all systems
        for system_id in self._systems:
            message = CoordinationMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.EMERGENCY_BROADCAST,
                sender_system="coordination_engine",
                recipient_system=system_id,
                timestamp=datetime.now(),
                payload={
                    "emergency_type": "system_emergency",
                    "event_payload": event.payload,
                    "timestamp": datetime.now().isoformat()
                },
                priority=3
            )
            
            self._send_message(message)
            
    def shutdown(self) -> None:
        """Shutdown coordination engine"""
        self.logger.info("Shutting down coordination engine")
        
        self._coordination_state = CoordinationState.DISCONNECTED
        self._shutdown_event.set()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear response waiters
        for future in self._response_waiters.values():
            if not future.done():
                future.set_exception(Exception("Coordination engine shutting down"))
                
        self.logger.info("Coordination engine shutdown complete")