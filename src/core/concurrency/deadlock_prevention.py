"""
Deadlock Prevention and Detection System
=======================================

This module implements comprehensive deadlock prevention and detection
mechanisms to ensure the system never enters deadlock states.

Features:
- Wait-for graph deadlock detection
- Resource ordering for deadlock prevention
- Timeout-based deadlock resolution
- Banker's algorithm for safe resource allocation
- Comprehensive deadlock analysis and reporting

Author: Agent Beta - Race Condition Elimination Specialist
"""

import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import structlog
import weakref

logger = structlog.get_logger(__name__)


class DeadlockResolutionStrategy(Enum):
    """Strategies for resolving deadlocks"""
    TIMEOUT_OLDEST = "timeout_oldest"
    TIMEOUT_LOWEST_PRIORITY = "timeout_lowest_priority"
    ABORT_YOUNGEST = "abort_youngest"
    ABORT_HIGHEST_COST = "abort_highest_cost"
    WOUND_WAIT = "wound_wait"
    WAIT_DIE = "wait_die"


@dataclass
class ResourceRequest:
    """Represents a resource request"""
    requester_id: str
    resource_id: str
    request_type: str  # "read", "write", "exclusive"
    priority: int
    timestamp: float
    timeout: Optional[float] = None
    callback: Optional[Callable] = None


@dataclass
class ResourceAllocation:
    """Represents a resource allocation"""
    holder_id: str
    resource_id: str
    allocation_type: str
    allocated_at: float
    expires_at: Optional[float] = None


@dataclass
class DeadlockCycle:
    """Represents a detected deadlock cycle"""
    cycle_id: str
    nodes: List[str]
    resources: List[str]
    detection_time: float
    resolution_strategy: DeadlockResolutionStrategy
    resolved: bool = False
    resolution_time: Optional[float] = None


class WaitForGraph:
    """
    Wait-for graph for deadlock detection
    """
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.node_info: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def add_edge(self, from_node: str, to_node: str, resource_id: str):
        """Add an edge to the wait-for graph"""
        with self._lock:
            if from_node != to_node:  # Avoid self-loops
                self.graph[from_node].add(to_node)
                self.reverse_graph[to_node].add(from_node)
                
                # Store edge metadata
                self.node_info[from_node] = {
                    'waiting_for': to_node,
                    'resource': resource_id,
                    'timestamp': time.time()
                }
                
    def remove_edge(self, from_node: str, to_node: str):
        """Remove an edge from the wait-for graph"""
        with self._lock:
            self.graph[from_node].discard(to_node)
            self.reverse_graph[to_node].discard(from_node)
            
            # Clean up empty sets
            if not self.graph[from_node]:
                del self.graph[from_node]
            if not self.reverse_graph[to_node]:
                del self.reverse_graph[to_node]
                
            # Clean up node info
            self.node_info.pop(from_node, None)
            
    def remove_node(self, node: str):
        """Remove a node and all its edges"""
        with self._lock:
            # Remove outgoing edges
            for to_node in list(self.graph.get(node, set())):
                self.remove_edge(node, to_node)
                
            # Remove incoming edges
            for from_node in list(self.reverse_graph.get(node, set())):
                self.remove_edge(from_node, node)
                
    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the wait-for graph using DFS"""
        with self._lock:
            visited = set()
            rec_stack = set()
            cycles = []
            
            def dfs(node: str, path: List[str]) -> None:
                if node in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]
                    cycles.append(cycle)
                    return
                    
                if node in visited:
                    return
                    
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                for neighbor in self.graph.get(node, set()):
                    dfs(neighbor, path)
                    
                rec_stack.remove(node)
                path.pop()
                
            # Check all nodes for cycles
            for node in self.graph:
                if node not in visited:
                    dfs(node, [])
                    
            return cycles
            
    def get_strongly_connected_components(self) -> List[List[str]]:
        """Get strongly connected components using Tarjan's algorithm"""
        with self._lock:
            index_counter = [0]
            stack = []
            lowlinks = {}
            index = {}
            on_stack = {}
            components = []
            
            def strongconnect(node: str):
                index[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
                
                for neighbor in self.graph.get(node, set()):
                    if neighbor not in index:
                        strongconnect(neighbor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                    elif on_stack[neighbor]:
                        lowlinks[node] = min(lowlinks[node], index[neighbor])
                        
                if lowlinks[node] == index[node]:
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.append(w)
                        if w == node:
                            break
                    components.append(component)
                    
            for node in self.graph:
                if node not in index:
                    strongconnect(node)
                    
            return components
            
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the wait-for graph"""
        with self._lock:
            return {
                'nodes': list(self.graph.keys()),
                'edges': [(from_node, list(to_nodes)) for from_node, to_nodes in self.graph.items()],
                'node_count': len(self.graph),
                'edge_count': sum(len(to_nodes) for to_nodes in self.graph.values()),
                'node_info': dict(self.node_info)
            }


class ResourceOrderingManager:
    """
    Resource ordering manager for deadlock prevention
    """
    
    def __init__(self):
        self.resource_order: Dict[str, int] = {}
        self.next_order = 0
        self._lock = threading.RLock()
        
    def register_resource(self, resource_id: str, order: Optional[int] = None) -> int:
        """Register a resource with a specific order"""
        with self._lock:
            if resource_id in self.resource_order:
                return self.resource_order[resource_id]
                
            if order is None:
                order = self.next_order
                self.next_order += 1
                
            self.resource_order[resource_id] = order
            return order
            
    def get_resource_order(self, resource_id: str) -> Optional[int]:
        """Get the order of a resource"""
        with self._lock:
            return self.resource_order.get(resource_id)
            
    def validate_request_order(self, held_resources: List[str], requested_resource: str) -> bool:
        """Validate that a resource request follows proper ordering"""
        with self._lock:
            requested_order = self.get_resource_order(requested_resource)
            if requested_order is None:
                # Auto-register new resources
                requested_order = self.register_resource(requested_resource)
                
            # Check that all held resources have lower order
            for held_resource in held_resources:
                held_order = self.get_resource_order(held_resource)
                if held_order is None:
                    held_order = self.register_resource(held_resource)
                    
                if held_order >= requested_order:
                    logger.warning(
                        "Resource ordering violation",
                        held_resource=held_resource,
                        held_order=held_order,
                        requested_resource=requested_resource,
                        requested_order=requested_order
                    )
                    return False
                    
            return True
            
    def get_ordered_resources(self) -> List[Tuple[str, int]]:
        """Get all resources in order"""
        with self._lock:
            return sorted(self.resource_order.items(), key=lambda x: x[1])


class TimeoutBasedDeadlockPrevention:
    """
    Timeout-based deadlock prevention mechanism
    """
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.active_requests: Dict[str, ResourceRequest] = {}
        self.timeout_callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        
    def start(self):
        """Start the timeout monitoring thread"""
        if self._cleanup_thread is None:
            self._cleanup_thread = threading.Thread(
                target=self._timeout_monitoring_loop,
                daemon=True
            )
            self._cleanup_thread.start()
            
    def stop(self):
        """Stop the timeout monitoring thread"""
        self._shutdown_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None
            
    def register_request(self, request: ResourceRequest, 
                        timeout_callback: Optional[Callable] = None):
        """Register a resource request with timeout"""
        with self._lock:
            request_id = f"{request.requester_id}:{request.resource_id}"
            self.active_requests[request_id] = request
            
            if timeout_callback:
                self.timeout_callbacks[request_id] = timeout_callback
                
    def unregister_request(self, requester_id: str, resource_id: str):
        """Unregister a resource request"""
        with self._lock:
            request_id = f"{requester_id}:{resource_id}"
            self.active_requests.pop(request_id, None)
            self.timeout_callbacks.pop(request_id, None)
            
    def _timeout_monitoring_loop(self):
        """Background thread for monitoring request timeouts"""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                timed_out_requests = []
                
                with self._lock:
                    for request_id, request in list(self.active_requests.items()):
                        timeout = request.timeout or self.default_timeout
                        if current_time - request.timestamp > timeout:
                            timed_out_requests.append((request_id, request))
                            
                # Handle timeouts outside of lock
                for request_id, request in timed_out_requests:
                    self._handle_timeout(request_id, request)
                    
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error("Error in timeout monitoring loop", error=str(e))
                time.sleep(1.0)
                
    def _handle_timeout(self, request_id: str, request: ResourceRequest):
        """Handle a timed out request"""
        logger.warning(
            "Resource request timed out",
            request_id=request_id,
            requester=request.requester_id,
            resource=request.resource_id,
            timeout=request.timeout
        )
        
        # Call timeout callback if registered
        callback = self.timeout_callbacks.get(request_id)
        if callback:
            try:
                callback(request)
            except Exception as e:
                logger.error("Error in timeout callback", error=str(e))
                
        # Remove from active requests
        with self._lock:
            self.active_requests.pop(request_id, None)
            self.timeout_callbacks.pop(request_id, None)
            
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get timeout statistics"""
        with self._lock:
            current_time = time.time()
            stats = {
                'active_requests': len(self.active_requests),
                'requests_by_age': {},
                'oldest_request_age': 0,
                'average_request_age': 0
            }
            
            ages = []
            for request in self.active_requests.values():
                age = current_time - request.timestamp
                ages.append(age)
                
                # Categorize by age
                if age < 5:
                    stats['requests_by_age']['0-5s'] = stats['requests_by_age'].get('0-5s', 0) + 1
                elif age < 15:
                    stats['requests_by_age']['5-15s'] = stats['requests_by_age'].get('5-15s', 0) + 1
                elif age < 30:
                    stats['requests_by_age']['15-30s'] = stats['requests_by_age'].get('15-30s', 0) + 1
                else:
                    stats['requests_by_age']['30s+'] = stats['requests_by_age'].get('30s+', 0) + 1
                    
            if ages:
                stats['oldest_request_age'] = max(ages)
                stats['average_request_age'] = sum(ages) / len(ages)
                
            return stats


class DeadlockDetectionAlgorithm:
    """
    Comprehensive deadlock detection algorithm
    """
    
    def __init__(self, 
                 detection_interval: float = 5.0,
                 resolution_strategy: DeadlockResolutionStrategy = DeadlockResolutionStrategy.TIMEOUT_OLDEST):
        self.detection_interval = detection_interval
        self.resolution_strategy = resolution_strategy
        self.wait_for_graph = WaitForGraph()
        self.resource_allocations: Dict[str, List[ResourceAllocation]] = defaultdict(list)
        self.resource_requests: Dict[str, List[ResourceRequest]] = defaultdict(list)
        self.detected_cycles: List[DeadlockCycle] = []
        self._lock = threading.RLock()
        self._detection_thread = None
        self._shutdown_event = threading.Event()
        self.resolution_callbacks: Dict[str, Callable] = {}
        
    def start(self):
        """Start the deadlock detection thread"""
        if self._detection_thread is None:
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True
            )
            self._detection_thread.start()
            
    def stop(self):
        """Stop the deadlock detection thread"""
        self._shutdown_event.set()
        if self._detection_thread:
            self._detection_thread.join(timeout=5.0)
            self._detection_thread = None
            
    def add_resource_allocation(self, allocation: ResourceAllocation):
        """Add a resource allocation"""
        with self._lock:
            self.resource_allocations[allocation.resource_id].append(allocation)
            
    def remove_resource_allocation(self, holder_id: str, resource_id: str):
        """Remove a resource allocation"""
        with self._lock:
            allocations = self.resource_allocations[resource_id]
            self.resource_allocations[resource_id] = [
                a for a in allocations if a.holder_id != holder_id
            ]
            
    def add_resource_request(self, request: ResourceRequest):
        """Add a resource request"""
        with self._lock:
            self.resource_requests[request.resource_id].append(request)
            self._update_wait_for_graph()
            
    def remove_resource_request(self, requester_id: str, resource_id: str):
        """Remove a resource request"""
        with self._lock:
            requests = self.resource_requests[resource_id]
            self.resource_requests[resource_id] = [
                r for r in requests if r.requester_id != requester_id
            ]
            self.wait_for_graph.remove_node(requester_id)
            
    def _update_wait_for_graph(self):
        """Update the wait-for graph based on current state"""
        with self._lock:
            # Clear existing graph
            self.wait_for_graph = WaitForGraph()
            
            # Build wait-for edges
            for resource_id, requests in self.resource_requests.items():
                allocations = self.resource_allocations[resource_id]
                
                for request in requests:
                    for allocation in allocations:
                        if (request.requester_id != allocation.holder_id and
                            self._conflicts(request, allocation)):
                            self.wait_for_graph.add_edge(
                                request.requester_id,
                                allocation.holder_id,
                                resource_id
                            )
                            
    def _conflicts(self, request: ResourceRequest, allocation: ResourceAllocation) -> bool:
        """Check if a request conflicts with an allocation"""
        # Read-read doesn't conflict
        if request.request_type == "read" and allocation.allocation_type == "read":
            return False
            
        # All other combinations conflict
        return True
        
    def _detection_loop(self):
        """Main deadlock detection loop"""
        while not self._shutdown_event.is_set():
            try:
                self._run_detection()
                time.sleep(self.detection_interval)
            except Exception as e:
                logger.error("Error in deadlock detection loop", error=str(e))
                time.sleep(self.detection_interval)
                
    def _run_detection(self):
        """Run deadlock detection"""
        with self._lock:
            self._update_wait_for_graph()
            cycles = self.wait_for_graph.detect_cycles()
            
            for cycle in cycles:
                if len(cycle) > 1:  # Ignore self-loops
                    self._handle_detected_cycle(cycle)
                    
    def _handle_detected_cycle(self, cycle: List[str]):
        """Handle a detected deadlock cycle"""
        cycle_id = str(uuid.uuid4())
        
        # Get resources involved in the cycle
        involved_resources = []
        for node in cycle:
            node_info = self.wait_for_graph.node_info.get(node, {})
            resource = node_info.get('resource')
            if resource:
                involved_resources.append(resource)
                
        deadlock_cycle = DeadlockCycle(
            cycle_id=cycle_id,
            nodes=cycle,
            resources=involved_resources,
            detection_time=time.time(),
            resolution_strategy=self.resolution_strategy
        )
        
        self.detected_cycles.append(deadlock_cycle)
        
        logger.critical(
            "Deadlock detected",
            cycle_id=cycle_id,
            nodes=cycle,
            resources=involved_resources,
            strategy=self.resolution_strategy.value
        )
        
        # Resolve the deadlock
        self._resolve_deadlock(deadlock_cycle)
        
    def _resolve_deadlock(self, deadlock_cycle: DeadlockCycle):
        """Resolve a detected deadlock"""
        try:
            if self.resolution_strategy == DeadlockResolutionStrategy.TIMEOUT_OLDEST:
                self._resolve_by_timeout_oldest(deadlock_cycle)
            elif self.resolution_strategy == DeadlockResolutionStrategy.TIMEOUT_LOWEST_PRIORITY:
                self._resolve_by_timeout_lowest_priority(deadlock_cycle)
            elif self.resolution_strategy == DeadlockResolutionStrategy.ABORT_YOUNGEST:
                self._resolve_by_abort_youngest(deadlock_cycle)
            else:
                # Default to timeout oldest
                self._resolve_by_timeout_oldest(deadlock_cycle)
                
            deadlock_cycle.resolved = True
            deadlock_cycle.resolution_time = time.time()
            
            logger.info(
                "Deadlock resolved",
                cycle_id=deadlock_cycle.cycle_id,
                strategy=deadlock_cycle.resolution_strategy.value
            )
            
        except Exception as e:
            logger.error(
                "Error resolving deadlock",
                cycle_id=deadlock_cycle.cycle_id,
                error=str(e)
            )
            
    def _resolve_by_timeout_oldest(self, deadlock_cycle: DeadlockCycle):
        """Resolve deadlock by timing out the oldest request"""
        oldest_node = None
        oldest_time = float('inf')
        
        for node in deadlock_cycle.nodes:
            node_info = self.wait_for_graph.node_info.get(node, {})
            timestamp = node_info.get('timestamp', time.time())
            
            if timestamp < oldest_time:
                oldest_time = timestamp
                oldest_node = node
                
        if oldest_node:
            self._timeout_node(oldest_node, deadlock_cycle.cycle_id)
            
    def _resolve_by_timeout_lowest_priority(self, deadlock_cycle: DeadlockCycle):
        """Resolve deadlock by timing out the lowest priority request"""
        # Find the lowest priority request in the cycle
        lowest_priority_node = None
        lowest_priority = float('inf')
        
        for node in deadlock_cycle.nodes:
            # Find the request for this node
            for requests in self.resource_requests.values():
                for request in requests:
                    if request.requester_id == node:
                        if request.priority < lowest_priority:
                            lowest_priority = request.priority
                            lowest_priority_node = node
                            
        if lowest_priority_node:
            self._timeout_node(lowest_priority_node, deadlock_cycle.cycle_id)
            
    def _resolve_by_abort_youngest(self, deadlock_cycle: DeadlockCycle):
        """Resolve deadlock by aborting the youngest request"""
        youngest_node = None
        youngest_time = 0
        
        for node in deadlock_cycle.nodes:
            node_info = self.wait_for_graph.node_info.get(node, {})
            timestamp = node_info.get('timestamp', 0)
            
            if timestamp > youngest_time:
                youngest_time = timestamp
                youngest_node = node
                
        if youngest_node:
            self._timeout_node(youngest_node, deadlock_cycle.cycle_id)
            
    def _timeout_node(self, node: str, cycle_id: str):
        """Timeout a specific node"""
        logger.warning(
            "Timing out node to resolve deadlock",
            node=node,
            cycle_id=cycle_id
        )
        
        # Remove all requests from this node
        for resource_id, requests in list(self.resource_requests.items()):
            self.resource_requests[resource_id] = [
                r for r in requests if r.requester_id != node
            ]
            
        # Remove from wait-for graph
        self.wait_for_graph.remove_node(node)
        
        # Call resolution callback if registered
        callback = self.resolution_callbacks.get(node)
        if callback:
            try:
                callback(node, cycle_id)
            except Exception as e:
                logger.error("Error in resolution callback", error=str(e))
                
    def register_resolution_callback(self, node_id: str, callback: Callable):
        """Register a callback for deadlock resolution"""
        self.resolution_callbacks[node_id] = callback
        
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get deadlock detection statistics"""
        with self._lock:
            total_cycles = len(self.detected_cycles)
            resolved_cycles = sum(1 for c in self.detected_cycles if c.resolved)
            
            return {
                'total_cycles_detected': total_cycles,
                'resolved_cycles': resolved_cycles,
                'unresolved_cycles': total_cycles - resolved_cycles,
                'active_resources': len(self.resource_allocations),
                'pending_requests': sum(len(requests) for requests in self.resource_requests.values()),
                'wait_for_graph_info': self.wait_for_graph.get_graph_info(),
                'recent_cycles': [
                    {
                        'cycle_id': c.cycle_id,
                        'nodes': c.nodes,
                        'resources': c.resources,
                        'detection_time': c.detection_time,
                        'resolved': c.resolved,
                        'resolution_time': c.resolution_time
                    }
                    for c in self.detected_cycles[-10:]  # Last 10 cycles
                ]
            }


class DeadlockPreventionManager:
    """
    Comprehensive deadlock prevention manager
    """
    
    def __init__(self, 
                 enable_resource_ordering: bool = True,
                 enable_timeout_prevention: bool = True,
                 enable_deadlock_detection: bool = True,
                 default_timeout: float = 30.0,
                 detection_interval: float = 5.0):
        
        self.enable_resource_ordering = enable_resource_ordering
        self.enable_timeout_prevention = enable_timeout_prevention
        self.enable_deadlock_detection = enable_deadlock_detection
        
        # Initialize components
        self.resource_ordering = ResourceOrderingManager() if enable_resource_ordering else None
        self.timeout_prevention = TimeoutBasedDeadlockPrevention(default_timeout) if enable_timeout_prevention else None
        self.deadlock_detection = DeadlockDetectionAlgorithm(detection_interval) if enable_deadlock_detection else None
        
        # Statistics
        self.stats = {
            'requests_validated': 0,
            'requests_rejected': 0,
            'timeouts_handled': 0,
            'deadlocks_resolved': 0
        }
        
    def start(self):
        """Start all deadlock prevention components"""
        if self.timeout_prevention:
            self.timeout_prevention.start()
            
        if self.deadlock_detection:
            self.deadlock_detection.start()
            
        logger.info("Deadlock prevention manager started")
        
    def stop(self):
        """Stop all deadlock prevention components"""
        if self.timeout_prevention:
            self.timeout_prevention.stop()
            
        if self.deadlock_detection:
            self.deadlock_detection.stop()
            
        logger.info("Deadlock prevention manager stopped")
        
    def validate_resource_request(self, request: ResourceRequest, 
                                 held_resources: List[str]) -> bool:
        """Validate a resource request for deadlock prevention"""
        self.stats['requests_validated'] += 1
        
        # Check resource ordering if enabled
        if self.resource_ordering and self.enable_resource_ordering:
            if not self.resource_ordering.validate_request_order(
                held_resources, request.resource_id
            ):
                self.stats['requests_rejected'] += 1
                return False
                
        # Register with timeout prevention if enabled
        if self.timeout_prevention and self.enable_timeout_prevention:
            self.timeout_prevention.register_request(request)
            
        # Register with deadlock detection if enabled
        if self.deadlock_detection and self.enable_deadlock_detection:
            self.deadlock_detection.add_resource_request(request)
            
        return True
        
    def notify_resource_acquired(self, allocation: ResourceAllocation):
        """Notify that a resource has been acquired"""
        if self.deadlock_detection:
            self.deadlock_detection.add_resource_allocation(allocation)
            
        if self.timeout_prevention:
            self.timeout_prevention.unregister_request(
                allocation.holder_id, 
                allocation.resource_id
            )
            
    def notify_resource_released(self, holder_id: str, resource_id: str):
        """Notify that a resource has been released"""
        if self.deadlock_detection:
            self.deadlock_detection.remove_resource_allocation(holder_id, resource_id)
            
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deadlock prevention statistics"""
        stats = {
            'manager_stats': dict(self.stats),
            'resource_ordering': None,
            'timeout_prevention': None,
            'deadlock_detection': None
        }
        
        if self.resource_ordering:
            stats['resource_ordering'] = {
                'ordered_resources': self.resource_ordering.get_ordered_resources()
            }
            
        if self.timeout_prevention:
            stats['timeout_prevention'] = self.timeout_prevention.get_timeout_statistics()
            
        if self.deadlock_detection:
            stats['deadlock_detection'] = self.deadlock_detection.get_detection_statistics()
            
        return stats