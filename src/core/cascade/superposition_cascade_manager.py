"""
SuperpositionCascadeManager - Core cascade orchestration system

This module provides the primary orchestration logic for managing the flow
of superpositions between Strategic → Tactical → Risk → Execution MARL systems.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from ..events import EventBus, Event, EventType
from ..errors import BaseException as CoreBaseException
from ..performance.performance_monitor import PerformanceMonitor
from ..resilience.circuit_breaker import CircuitBreaker
from ..resilience.retry_manager import RetryManager


class CascadeState(Enum):
    """Cascade system states"""
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    FLOWING = "FLOWING"
    DEGRADED = "DEGRADED"
    EMERGENCY = "EMERGENCY"
    SHUTDOWN = "SHUTDOWN"


class SuperpositionType(Enum):
    """Types of superposition data flowing through cascade"""
    STRATEGIC_SIGNAL = "STRATEGIC_SIGNAL"
    TACTICAL_SIGNAL = "TACTICAL_SIGNAL"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    EXECUTION_PLAN = "EXECUTION_PLAN"
    CONTEXT_UPDATE = "CONTEXT_UPDATE"
    EMERGENCY_SIGNAL = "EMERGENCY_SIGNAL"


@dataclass
class SuperpositionPacket:
    """Standardized superposition data packet"""
    packet_id: str
    packet_type: SuperpositionType
    source_system: str
    target_system: str
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any]
    priority: int = 1  # 1=normal, 2=high, 3=emergency
    ttl: Optional[datetime] = None  # Time to live
    retry_count: int = 0
    max_retries: int = 3
    

@dataclass
class CascadeMetrics:
    """Cascade performance metrics"""
    total_packets_processed: int = 0
    packets_per_second: float = 0.0
    average_latency_ms: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    current_queue_depth: int = 0
    system_load: float = 0.0
    cascade_health_score: float = 100.0


class SuperpositionCascadeManager:
    """
    Core cascade orchestration system that manages the flow of superpositions
    between Strategic → Tactical → Risk → Execution MARL systems.
    """

    def __init__(
        self,
        event_bus: EventBus,
        performance_monitor: Optional[PerformanceMonitor] = None,
        max_concurrent_flows: int = 100,
        cascade_timeout_ms: int = 100,
        emergency_threshold: float = 0.8
    ):
        self.event_bus = event_bus
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.max_concurrent_flows = max_concurrent_flows
        self.cascade_timeout_ms = cascade_timeout_ms
        self.emergency_threshold = emergency_threshold
        
        # State management
        self.state = CascadeState.INITIALIZING
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Threading and concurrency
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_flows)
        self._shutdown_event = threading.Event()
        
        # Queue management
        self._packet_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._priority_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._emergency_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        
        # System registry
        self._marl_systems: Dict[str, Dict[str, Any]] = {}
        self._cascade_chain = ["strategic", "tactical", "risk", "execution"]
        
        # Performance tracking
        self._metrics = CascadeMetrics()
        self._latency_buffer: List[float] = []
        self._start_time = time.time()
        
        # Resilience components
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retry_manager = RetryManager()
        
        # Callbacks
        self._flow_callbacks: Dict[str, List[Callable]] = {}
        self._error_callbacks: List[Callable] = []
        
        # Initialize components
        self._initialize_cascade_system()
        
    def _initialize_cascade_system(self) -> None:
        """Initialize the cascade management system"""
        try:
            # Register for relevant events
            self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_strategic_decision)
            self.event_bus.subscribe(EventType.TACTICAL_DECISION, self._handle_tactical_decision)
            self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
            self.event_bus.subscribe(EventType.EXECUTE_TRADE, self._handle_execution_request)
            self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
            
            # Initialize circuit breakers for each system
            for system_name in self._cascade_chain:
                self._circuit_breakers[system_name] = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=30,
                    expected_exception=CoreBaseException
                )
                
            # Initialize MARL system registry
            self._initialize_marl_systems()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.state = CascadeState.READY
            self.logger.info("Cascade system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cascade system: {e}")
            self.state = CascadeState.EMERGENCY
            raise
            
    def _initialize_marl_systems(self) -> None:
        """Initialize MARL system registry"""
        system_configs = {
            "strategic": {
                "name": "Strategic MARL",
                "input_types": [SuperpositionType.CONTEXT_UPDATE],
                "output_types": [SuperpositionType.STRATEGIC_SIGNAL],
                "timeout_ms": 30,
                "max_retries": 2
            },
            "tactical": {
                "name": "Tactical MARL", 
                "input_types": [SuperpositionType.STRATEGIC_SIGNAL],
                "output_types": [SuperpositionType.TACTICAL_SIGNAL],
                "timeout_ms": 20,
                "max_retries": 2
            },
            "risk": {
                "name": "Risk Management MARL",
                "input_types": [SuperpositionType.TACTICAL_SIGNAL],
                "output_types": [SuperpositionType.RISK_ASSESSMENT],
                "timeout_ms": 25,
                "max_retries": 3
            },
            "execution": {
                "name": "Execution MARL",
                "input_types": [SuperpositionType.RISK_ASSESSMENT],
                "output_types": [SuperpositionType.EXECUTION_PLAN],
                "timeout_ms": 25,
                "max_retries": 2
            }
        }
        
        for system_id, config in system_configs.items():
            self._marl_systems[system_id] = {
                **config,
                "status": "active",
                "last_update": datetime.now(),
                "error_count": 0,
                "success_count": 0
            }
            
    def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        threading.Thread(target=self._packet_processor, daemon=True).start()
        threading.Thread(target=self._metrics_updater, daemon=True).start()
        threading.Thread(target=self._health_monitor, daemon=True).start()
        
    def register_marl_system(
        self,
        system_id: str,
        system_name: str,
        input_callback: Callable[[SuperpositionPacket], Optional[SuperpositionPacket]],
        output_callback: Optional[Callable[[SuperpositionPacket], None]] = None
    ) -> None:
        """Register a MARL system with the cascade"""
        with self._lock:
            if system_id in self._marl_systems:
                self._marl_systems[system_id].update({
                    "input_callback": input_callback,
                    "output_callback": output_callback,
                    "status": "active"
                })
                self.logger.info(f"MARL system registered: {system_name}")
            else:
                self.logger.warning(f"Unknown MARL system: {system_id}")
                
    def inject_superposition(
        self,
        packet_type: SuperpositionType,
        data: Dict[str, Any],
        source_system: str = "external",
        priority: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Inject a superposition packet into the cascade flow"""
        packet_id = f"{source_system}_{int(time.time() * 1000000)}"
        
        # Determine target system based on packet type
        target_system = self._determine_target_system(packet_type, source_system)
        
        packet = SuperpositionPacket(
            packet_id=packet_id,
            packet_type=packet_type,
            source_system=source_system,
            target_system=target_system,
            timestamp=datetime.now(),
            data=data,
            context=context or {},
            priority=priority,
            ttl=datetime.now() + timedelta(seconds=30)
        )
        
        # Queue packet based on priority
        try:
            if priority == 3:  # Emergency
                self._emergency_queue.put_nowait(packet)
            elif priority == 2:  # High
                self._priority_queue.put_nowait(packet)
            else:  # Normal
                self._packet_queue.put_nowait(packet)
                
            self.logger.debug(f"Superposition packet queued: {packet_id}")
            return packet_id
            
        except asyncio.QueueFull:
            self.logger.error(f"Queue full, dropping packet: {packet_id}")
            return None
            
    def _determine_target_system(self, packet_type: SuperpositionType, source_system: str) -> str:
        """Determine target system based on packet type and source"""
        if packet_type == SuperpositionType.STRATEGIC_SIGNAL:
            return "tactical"
        elif packet_type == SuperpositionType.TACTICAL_SIGNAL:
            return "risk"
        elif packet_type == SuperpositionType.RISK_ASSESSMENT:
            return "execution"
        elif packet_type == SuperpositionType.CONTEXT_UPDATE:
            return "strategic"
        else:
            # Default to next system in chain
            if source_system in self._cascade_chain:
                idx = self._cascade_chain.index(source_system)
                if idx < len(self._cascade_chain) - 1:
                    return self._cascade_chain[idx + 1]
            return "strategic"  # Default fallback
            
    def _packet_processor(self) -> None:
        """Background packet processing thread"""
        self.logger.info("Packet processor started")
        
        while not self._shutdown_event.is_set():
            try:
                # Process emergency packets first
                packet = self._get_next_packet()
                if packet:
                    self._process_packet(packet)
                else:
                    time.sleep(0.001)  # Brief pause if no packets
                    
            except Exception as e:
                self.logger.error(f"Error in packet processor: {e}")
                time.sleep(0.01)
                
    def _get_next_packet(self) -> Optional[SuperpositionPacket]:
        """Get next packet to process (priority-based)"""
        try:
            # Emergency queue first
            if not self._emergency_queue.empty():
                return self._emergency_queue.get_nowait()
            # High priority queue
            elif not self._priority_queue.empty():
                return self._priority_queue.get_nowait()
            # Normal queue
            elif not self._packet_queue.empty():
                return self._packet_queue.get_nowait()
            else:
                return None
        except asyncio.QueueEmpty:
            return None
            
    def _process_packet(self, packet: SuperpositionPacket) -> None:
        """Process a single superposition packet"""
        start_time = time.time()
        
        try:
            # Check TTL
            if packet.ttl and datetime.now() > packet.ttl:
                self.logger.warning(f"Packet TTL expired: {packet.packet_id}")
                return
                
            # Get target system
            target_system = self._marl_systems.get(packet.target_system)
            if not target_system:
                self.logger.error(f"Unknown target system: {packet.target_system}")
                return
                
            # Check circuit breaker
            circuit_breaker = self._circuit_breakers.get(packet.target_system)
            if circuit_breaker and circuit_breaker.state == "open":
                self.logger.warning(f"Circuit breaker open for {packet.target_system}")
                self._handle_circuit_breaker_open(packet)
                return
                
            # Process packet through target system
            result = self._execute_system_processing(packet, target_system)
            
            if result:
                # Forward result to next system if applicable
                self._forward_result(result)
                
                # Update success metrics
                target_system["success_count"] += 1
                self._update_latency(time.time() - start_time)
                
            else:
                # Handle processing failure
                self._handle_processing_failure(packet, target_system)
                
        except Exception as e:
            self.logger.error(f"Error processing packet {packet.packet_id}: {e}")
            self._handle_processing_error(packet, e)
            
    def _execute_system_processing(
        self, 
        packet: SuperpositionPacket,
        target_system: Dict[str, Any]
    ) -> Optional[SuperpositionPacket]:
        """Execute processing by target MARL system"""
        try:
            # Get system callback
            input_callback = target_system.get("input_callback")
            if not input_callback:
                self.logger.error(f"No input callback for system: {packet.target_system}")
                return None
                
            # Execute with timeout
            future = self._executor.submit(input_callback, packet)
            timeout = target_system.get("timeout_ms", 50) / 1000.0
            
            result = future.result(timeout=timeout)
            
            # Update system status
            target_system["last_update"] = datetime.now()
            target_system["status"] = "active"
            
            return result
            
        except TimeoutError:
            self.logger.error(f"Timeout processing packet in {packet.target_system}")
            return None
        except Exception as e:
            self.logger.error(f"Error in system processing: {e}")
            return None
            
    def _forward_result(self, result: SuperpositionPacket) -> None:
        """Forward processing result to next system"""
        try:
            # Determine next system in cascade
            next_system = self._determine_target_system(result.packet_type, result.source_system)
            
            if next_system and next_system in self._marl_systems:
                result.target_system = next_system
                result.timestamp = datetime.now()
                
                # Re-queue for processing
                if result.priority == 3:
                    self._emergency_queue.put_nowait(result)
                elif result.priority == 2:
                    self._priority_queue.put_nowait(result)
                else:
                    self._packet_queue.put_nowait(result)
                    
        except Exception as e:
            self.logger.error(f"Error forwarding result: {e}")
            
    def _handle_processing_failure(
        self, 
        packet: SuperpositionPacket,
        target_system: Dict[str, Any]
    ) -> None:
        """Handle packet processing failure"""
        packet.retry_count += 1
        target_system["error_count"] += 1
        
        if packet.retry_count < packet.max_retries:
            # Retry with exponential backoff
            delay = 0.001 * (2 ** packet.retry_count)
            threading.Timer(delay, lambda: self._requeue_packet(packet)).start()
            self.logger.warning(f"Retrying packet {packet.packet_id} (attempt {packet.retry_count})")
        else:
            # Max retries exceeded
            self.logger.error(f"Max retries exceeded for packet {packet.packet_id}")
            self._handle_dead_letter(packet)
            
    def _handle_processing_error(self, packet: SuperpositionPacket, error: Exception) -> None:
        """Handle processing errors"""
        self.logger.error(f"Processing error for packet {packet.packet_id}: {error}")
        
        # Notify error callbacks
        for callback in self._error_callbacks:
            try:
                callback(packet, error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
                
    def _handle_circuit_breaker_open(self, packet: SuperpositionPacket) -> None:
        """Handle circuit breaker open state"""
        if packet.priority == 3:  # Emergency packets bypass circuit breaker
            self.logger.warning(f"Emergency packet bypassing circuit breaker: {packet.packet_id}")
            self._process_packet(packet)
        else:
            self.logger.warning(f"Packet dropped due to circuit breaker: {packet.packet_id}")
            self._handle_dead_letter(packet)
            
    def _handle_dead_letter(self, packet: SuperpositionPacket) -> None:
        """Handle packets that cannot be processed"""
        self.logger.error(f"Dead letter: {packet.packet_id}")
        
        # Publish dead letter event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.SYSTEM_ERROR,
                {
                    "type": "dead_letter",
                    "packet_id": packet.packet_id,
                    "packet_type": packet.packet_type.value,
                    "target_system": packet.target_system
                },
                "cascade_manager"
            )
        )
        
    def _requeue_packet(self, packet: SuperpositionPacket) -> None:
        """Requeue packet for retry"""
        try:
            if packet.priority == 3:
                self._emergency_queue.put_nowait(packet)
            elif packet.priority == 2:
                self._priority_queue.put_nowait(packet)
            else:
                self._packet_queue.put_nowait(packet)
        except asyncio.QueueFull:
            self.logger.error(f"Queue full during requeue: {packet.packet_id}")
            self._handle_dead_letter(packet)
            
    def _metrics_updater(self) -> None:
        """Background metrics update thread"""
        while not self._shutdown_event.is_set():
            try:
                self._update_metrics()
                time.sleep(1.0)  # Update every second
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                
    def _update_metrics(self) -> None:
        """Update cascade metrics"""
        with self._lock:
            # Calculate current metrics
            current_time = time.time()
            elapsed = current_time - self._start_time
            
            # Queue depths
            self._metrics.current_queue_depth = (
                self._packet_queue.qsize() + 
                self._priority_queue.qsize() + 
                self._emergency_queue.qsize()
            )
            
            # Calculate average latency
            if self._latency_buffer:
                self._metrics.average_latency_ms = sum(self._latency_buffer) / len(self._latency_buffer) * 1000
                
                # Keep only recent latencies
                if len(self._latency_buffer) > 1000:
                    self._latency_buffer = self._latency_buffer[-1000:]
                    
            # Calculate success rate
            total_success = sum(system["success_count"] for system in self._marl_systems.values())
            total_errors = sum(system["error_count"] for system in self._marl_systems.values())
            total_ops = total_success + total_errors
            
            if total_ops > 0:
                self._metrics.success_rate = (total_success / total_ops) * 100
                self._metrics.error_rate = (total_errors / total_ops) * 100
                
            # Calculate throughput
            self._metrics.packets_per_second = total_ops / elapsed if elapsed > 0 else 0
            
            # Calculate health score
            self._metrics.cascade_health_score = self._calculate_health_score()
            
            # Check for emergency state
            if self._metrics.cascade_health_score < self.emergency_threshold * 100:
                self._trigger_emergency_state()
                
    def _calculate_health_score(self) -> float:
        """Calculate overall cascade health score"""
        factors = []
        
        # Success rate factor
        factors.append(self._metrics.success_rate)
        
        # Latency factor (inverse)
        latency_factor = max(0, 100 - (self._metrics.average_latency_ms / 10))
        factors.append(latency_factor)
        
        # Queue depth factor
        queue_factor = max(0, 100 - (self._metrics.current_queue_depth / 10))
        factors.append(queue_factor)
        
        # System availability factor
        active_systems = sum(1 for system in self._marl_systems.values() if system["status"] == "active")
        availability_factor = (active_systems / len(self._marl_systems)) * 100
        factors.append(availability_factor)
        
        # Calculate weighted average
        return sum(factors) / len(factors)
        
    def _update_latency(self, latency_seconds: float) -> None:
        """Update latency tracking"""
        self._latency_buffer.append(latency_seconds)
        
    def _health_monitor(self) -> None:
        """Background health monitoring thread"""
        while not self._shutdown_event.is_set():
            try:
                self._check_system_health()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                
    def _check_system_health(self) -> None:
        """Check health of all MARL systems"""
        current_time = datetime.now()
        
        for system_id, system in self._marl_systems.items():
            last_update = system["last_update"]
            time_since_update = (current_time - last_update).total_seconds()
            
            # Check for stale systems
            if time_since_update > 60:  # 1 minute
                system["status"] = "stale"
                self.logger.warning(f"System {system_id} is stale")
                
            # Check error rates
            error_rate = system["error_count"] / max(1, system["success_count"] + system["error_count"])
            if error_rate > 0.1:  # 10% error rate
                system["status"] = "degraded"
                self.logger.warning(f"System {system_id} has high error rate: {error_rate:.2%}")
                
    def _trigger_emergency_state(self) -> None:
        """Trigger emergency state"""
        if self.state != CascadeState.EMERGENCY:
            self.state = CascadeState.EMERGENCY
            self.logger.critical("CASCADE EMERGENCY STATE TRIGGERED")
            
            # Publish emergency event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        "source": "cascade_manager",
                        "reason": "cascade_health_degraded",
                        "health_score": self._metrics.cascade_health_score
                    },
                    "cascade_manager"
                )
            )
            
    def get_cascade_status(self) -> Dict[str, Any]:
        """Get current cascade status"""
        return {
            "state": self.state.value,
            "metrics": {
                "total_packets_processed": self._metrics.total_packets_processed,
                "packets_per_second": self._metrics.packets_per_second,
                "average_latency_ms": self._metrics.average_latency_ms,
                "success_rate": self._metrics.success_rate,
                "error_rate": self._metrics.error_rate,
                "current_queue_depth": self._metrics.current_queue_depth,
                "cascade_health_score": self._metrics.cascade_health_score
            },
            "systems": {
                system_id: {
                    "status": system["status"],
                    "success_count": system["success_count"],
                    "error_count": system["error_count"],
                    "last_update": system["last_update"].isoformat()
                }
                for system_id, system in self._marl_systems.items()
            }
        }
        
    def get_metrics(self) -> CascadeMetrics:
        """Get cascade metrics"""
        return self._metrics
        
    def add_flow_callback(self, flow_type: str, callback: Callable) -> None:
        """Add callback for specific flow type"""
        if flow_type not in self._flow_callbacks:
            self._flow_callbacks[flow_type] = []
        self._flow_callbacks[flow_type].append(callback)
        
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback"""
        self._error_callbacks.append(callback)
        
    def emergency_shutdown(self) -> None:
        """Emergency shutdown of cascade system"""
        self.logger.critical("Emergency shutdown initiated")
        self.state = CascadeState.SHUTDOWN
        self._shutdown_event.set()
        
        # Clear all queues
        self._clear_all_queues()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
    def _clear_all_queues(self) -> None:
        """Clear all packet queues"""
        queues = [self._packet_queue, self._priority_queue, self._emergency_queue]
        
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        self.logger.info("Graceful shutdown initiated")
        self.state = CascadeState.SHUTDOWN
        self._shutdown_event.set()
        
        # Wait for queues to drain
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while (
            not self._packet_queue.empty() or 
            not self._priority_queue.empty() or 
            not self._emergency_queue.empty()
        ):
            if time.time() - start_time > timeout:
                self.logger.warning("Shutdown timeout reached, forcing shutdown")
                break
            time.sleep(0.1)
            
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Cascade manager shutdown complete")
        
    # Event handlers
    def _handle_strategic_decision(self, event: Event) -> None:
        """Handle strategic decision events"""
        self.inject_superposition(
            SuperpositionType.STRATEGIC_SIGNAL,
            event.payload,
            source_system="strategic",
            context={"event_id": id(event)}
        )
        
    def _handle_tactical_decision(self, event: Event) -> None:
        """Handle tactical decision events"""
        self.inject_superposition(
            SuperpositionType.TACTICAL_SIGNAL,
            event.payload,
            source_system="tactical",
            context={"event_id": id(event)}
        )
        
    def _handle_risk_update(self, event: Event) -> None:
        """Handle risk update events"""
        self.inject_superposition(
            SuperpositionType.RISK_ASSESSMENT,
            event.payload,
            source_system="risk",
            context={"event_id": id(event)}
        )
        
    def _handle_execution_request(self, event: Event) -> None:
        """Handle execution request events"""
        self.inject_superposition(
            SuperpositionType.EXECUTION_PLAN,
            event.payload,
            source_system="execution",
            priority=2,  # High priority
            context={"event_id": id(event)}
        )
        
    def _handle_emergency_stop(self, event: Event) -> None:
        """Handle emergency stop events"""
        self.inject_superposition(
            SuperpositionType.EMERGENCY_SIGNAL,
            event.payload,
            source_system="emergency",
            priority=3,  # Emergency priority
            context={"event_id": id(event)}
        )