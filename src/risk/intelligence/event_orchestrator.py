"""
Advanced Event Processing System - Intelligence Integration Layer

Complex Event Processing (CEP) engine for multi-agent coordination with priority 
management, event correlation, and real-time stream processing.

Features:
- Priority-based event routing and scheduling
- Complex event pattern detection across agents
- Real-time event stream processing with <5ms latency
- Event correlation and causality analysis
- Event replay and debugging capabilities
- Intelligent event filtering and aggregation

Architecture:
- Event Ingestion: High-throughput event collection from all agents
- Event Classification: Priority-based event categorization
- Event Correlation: Pattern detection across event streams
- Event Routing: Intelligent routing to appropriate handlers
- Event Replay: Historical event reconstruction for debugging
"""

import asyncio
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import defaultdict, deque
import heapq
import weakref
import json
import re
from abc import ABC, abstractmethod

from src.core.events import EventBus, Event, EventType
from src.risk.intelligence.intelligence_coordinator import IntelligencePriority

logger = structlog.get_logger()


class EventPriority(IntEnum):
    """Event priority levels (lower number = higher priority)"""
    EMERGENCY = 0      # Crisis detection, emergency stops
    CRITICAL = 1       # System failures, risk breaches
    HIGH = 2          # Human decisions, important alerts
    MEDIUM = 3        # Trade signals, pre-mortem analysis
    LOW = 4           # Performance monitoring, logging
    BACKGROUND = 5    # Maintenance, cleanup tasks


class EventCategory(Enum):
    """Event categories for classification"""
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    RISK_ALERT = "risk_alert"
    SYSTEM_HEALTH = "system_health"
    INTELLIGENCE = "intelligence"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"


class EventPattern(Enum):
    """Complex event patterns for detection"""
    CORRELATION_SPIKE = "correlation_spike"
    CASCADING_FAILURE = "cascading_failure"
    TRADING_ANOMALY = "trading_anomaly"
    SYSTEM_DEGRADATION = "system_degradation"
    CRISIS_SEQUENCE = "crisis_sequence"
    HUMAN_INTERVENTION = "human_intervention"


@dataclass
class EventMetadata:
    """Enhanced event metadata"""
    event_id: str
    sequence_number: int
    priority: EventPriority
    category: EventCategory
    source_component: str
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    processing_deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ProcessedEvent:
    """Event with processing metadata"""
    original_event: Event
    metadata: EventMetadata
    processing_start: datetime
    processing_end: Optional[datetime] = None
    handler_results: Dict[str, Any] = field(default_factory=dict)
    correlation_matches: List[str] = field(default_factory=list)
    pattern_matches: List[EventPattern] = field(default_factory=list)


@dataclass
class EventCorrelation:
    """Event correlation definition"""
    pattern_id: str
    pattern_type: EventPattern
    event_types: List[EventType]
    time_window_ms: int
    correlation_threshold: float
    handler: Callable[[List[ProcessedEvent]], None]


@dataclass
class EventRoute:
    """Event routing configuration"""
    event_type: EventType
    priority: EventPriority
    category: EventCategory
    handlers: List[Callable[[ProcessedEvent], None]]
    max_processing_time_ms: float
    requires_ordering: bool = False


class EventPatternMatcher:
    """Complex event pattern detection engine"""
    
    def __init__(self, time_window_ms: int = 10000):
        self.time_window_ms = time_window_ms
        self.event_history: deque = deque(maxlen=1000)
        self.pattern_rules: Dict[EventPattern, EventCorrelation] = {}
        self.active_patterns: Dict[str, List[ProcessedEvent]] = defaultdict(list)
    
    def register_pattern(self, correlation: EventCorrelation):
        """Register a complex event pattern"""
        self.pattern_rules[correlation.pattern_type] = correlation
        logger.debug("Event pattern registered", 
                    pattern=correlation.pattern_type.value,
                    event_types=[et.value for et in correlation.event_types])
    
    def detect_patterns(self, event: ProcessedEvent) -> List[EventPattern]:
        """Detect complex patterns in event stream"""
        self.event_history.append(event)
        detected_patterns = []
        
        current_time = event.processing_start
        window_start = current_time - timedelta(milliseconds=self.time_window_ms)
        
        # Get recent events within time window
        recent_events = [e for e in self.event_history 
                        if e.processing_start >= window_start]
        
        # Check each pattern rule
        for pattern_type, correlation in self.pattern_rules.items():
            if self._check_pattern_match(recent_events, correlation):
                detected_patterns.append(pattern_type)
                self._trigger_pattern_handler(pattern_type, recent_events, correlation)
        
        return detected_patterns
    
    def _check_pattern_match(self, events: List[ProcessedEvent], correlation: EventCorrelation) -> bool:
        """Check if events match a pattern"""
        # Group events by type
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.original_event.event_type].append(event)
        
        # Check if all required event types are present
        required_types = set(correlation.event_types)
        available_types = set(events_by_type.keys())
        
        if not required_types.issubset(available_types):
            return False
        
        # Pattern-specific correlation logic
        if correlation.pattern_type == EventPattern.CORRELATION_SPIKE:
            return self._check_correlation_spike(events_by_type, correlation)
        elif correlation.pattern_type == EventPattern.CASCADING_FAILURE:
            return self._check_cascading_failure(events_by_type, correlation)
        elif correlation.pattern_type == EventPattern.CRISIS_SEQUENCE:
            return self._check_crisis_sequence(events_by_type, correlation)
        
        return False
    
    def _check_correlation_spike(self, events_by_type: Dict, correlation: EventCorrelation) -> bool:
        """Check for correlation spike pattern"""
        # Look for multiple VAR updates or risk alerts in short time
        risk_events = events_by_type.get(EventType.VAR_UPDATE, []) + \
                     events_by_type.get(EventType.RISK_UPDATE, [])
        
        return len(risk_events) >= 3  # 3 or more risk events
    
    def _check_cascading_failure(self, events_by_type: Dict, correlation: EventCorrelation) -> bool:
        """Check for cascading failure pattern"""
        # Look for sequence: system error -> risk breach -> emergency stop
        has_error = bool(events_by_type.get(EventType.SYSTEM_ERROR, []))
        has_risk = bool(events_by_type.get(EventType.RISK_BREACH, []))
        has_emergency = bool(events_by_type.get(EventType.EMERGENCY_STOP, []))
        
        return has_error and has_risk and has_emergency
    
    def _check_crisis_sequence(self, events_by_type: Dict, correlation: EventCorrelation) -> bool:
        """Check for crisis sequence pattern"""
        # Look for market stress -> risk alerts -> emergency actions
        has_stress = bool(events_by_type.get(EventType.MARKET_STRESS, []))
        has_risk = bool(events_by_type.get(EventType.RISK_BREACH, []))
        has_trades = bool(events_by_type.get(EventType.TRADE_REJECTED, []))
        
        return has_stress and has_risk and has_trades
    
    def _trigger_pattern_handler(self, pattern_type: EventPattern, 
                                events: List[ProcessedEvent], 
                                correlation: EventCorrelation):
        """Trigger pattern-specific handler"""
        try:
            matching_events = [e for e in events 
                             if e.original_event.event_type in correlation.event_types]
            correlation.handler(matching_events)
            logger.info("Complex event pattern detected", 
                       pattern=pattern_type.value,
                       event_count=len(matching_events))
        except Exception as e:
            logger.error("Error in pattern handler", 
                        pattern=pattern_type.value, error=str(e))


class EventOrchestrator:
    """
    Advanced Event Processing System for Intelligence Integration
    
    Provides complex event processing with priority management, correlation detection,
    and real-time stream processing for coordinating all intelligence components.
    """
    
    def __init__(self, max_throughput_events_per_sec: int = 10000):
        self.max_throughput = max_throughput_events_per_sec
        
        # Event processing infrastructure
        self.event_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.thread_count = 4
        self.running = False
        
        # Event classification and routing
        self.event_routes: Dict[EventType, EventRoute] = {}
        self.event_handlers: Dict[str, Callable] = {}
        self.default_handlers: List[Callable] = []
        
        # Event correlation and pattern detection
        self.pattern_matcher = EventPatternMatcher()
        self.correlation_engine = {}
        
        # Performance tracking
        self.event_count = 0
        self.processing_times: deque = deque(maxlen=1000)
        self.throughput_history: deque = deque(maxlen=60)  # 1 minute history
        self.error_count = 0
        
        # Event history for replay and debugging
        self.event_history: deque = deque(maxlen=10000)
        self.processed_events: deque = deque(maxlen=5000)
        
        # Sequence tracking
        self.sequence_counter = 0
        self.sequence_lock = threading.Lock()
        
        # Quality metrics
        self.latency_target_ms = 5.0
        self.latency_violations = 0
        
        logger.info("Event orchestrator initialized", 
                   target_throughput=max_throughput_events_per_sec,
                   thread_count=self.thread_count)
    
    def register_event_route(self, 
                           event_type: EventType,
                           priority: EventPriority,
                           category: EventCategory,
                           handlers: List[Callable],
                           max_processing_time_ms: float = 5.0,
                           requires_ordering: bool = False):
        """Register event routing configuration"""
        route = EventRoute(
            event_type=event_type,
            priority=priority,
            category=category,
            handlers=handlers,
            max_processing_time_ms=max_processing_time_ms,
            requires_ordering=requires_ordering
        )
        
        self.event_routes[event_type] = route
        logger.debug("Event route registered",
                    event_type=event_type.value,
                    priority=priority.value,
                    category=category.value,
                    handler_count=len(handlers))
    
    def register_pattern_correlation(self, correlation: EventCorrelation):
        """Register complex event pattern for detection"""
        self.pattern_matcher.register_pattern(correlation)
    
    def submit_event(self, event: Event, 
                    correlation_id: Optional[str] = None,
                    parent_event_id: Optional[str] = None,
                    processing_deadline: Optional[datetime] = None) -> str:
        """Submit event for processing"""
        
        # Generate event metadata
        with self.sequence_lock:
            self.sequence_counter += 1
            sequence_number = self.sequence_counter
        
        event_id = f"evt_{sequence_number}_{int(time.time() * 1000000)}"
        
        # Determine priority and category from route
        route = self.event_routes.get(event.event_type)
        if route:
            priority = route.priority
            category = route.category
        else:
            priority = self._infer_event_priority(event.event_type)
            category = self._infer_event_category(event.event_type)
        
        metadata = EventMetadata(
            event_id=event_id,
            sequence_number=sequence_number,
            priority=priority,
            category=category,
            source_component=event.source,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            processing_deadline=processing_deadline or 
                              (datetime.now() + timedelta(milliseconds=self.latency_target_ms))
        )
        
        # Create processed event wrapper
        processed_event = ProcessedEvent(
            original_event=event,
            metadata=metadata,
            processing_start=datetime.now()
        )
        
        # Submit to priority queue
        try:
            self.event_queue.put((priority.value, sequence_number, processed_event), timeout=0.001)
            self.event_count += 1
            
            # Add to history
            self.event_history.append(processed_event)
            
            return event_id
            
        except queue.Full:
            logger.error("Event queue full, dropping event",
                        event_type=event.event_type.value,
                        queue_size=self.event_queue.qsize())
            self.error_count += 1
            return ""
    
    def start_processing(self):
        """Start event processing threads"""
        if self.running:
            return
        
        self.running = True
        
        # Start processing threads
        for i in range(self.thread_count):
            thread = threading.Thread(
                target=self._event_processing_loop,
                name=f"event_processor_{i}"
            )
            thread.start()
            self.processing_threads.append(thread)
        
        # Start throughput monitoring
        monitor_thread = threading.Thread(
            target=self._throughput_monitor_loop,
            name="throughput_monitor"
        )
        monitor_thread.start()
        self.processing_threads.append(monitor_thread)
        
        logger.info("Event processing started", thread_count=self.thread_count)
    
    def stop_processing(self):
        """Stop event processing"""
        self.running = False
        
        # Wait for threads to complete
        for thread in self.processing_threads:
            thread.join(timeout=2.0)
        
        self.processing_threads.clear()
        logger.info("Event processing stopped")
    
    def _event_processing_loop(self):
        """Main event processing loop"""
        while self.running:
            try:
                # Get next event from priority queue
                try:
                    priority, sequence, processed_event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the event
                self._process_event(processed_event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error("Error in event processing loop", error=str(e))
                self.error_count += 1
                time.sleep(0.001)  # Brief pause on error
    
    def _process_event(self, processed_event: ProcessedEvent):
        """Process individual event"""
        start_time = datetime.now()
        
        try:
            # Check processing deadline
            if processed_event.metadata.processing_deadline and \
               start_time > processed_event.metadata.processing_deadline:
                logger.warning("Event processing deadline exceeded",
                             event_id=processed_event.metadata.event_id,
                             delay_ms=(start_time - processed_event.metadata.processing_deadline).total_seconds() * 1000)
                self.latency_violations += 1
            
            # Detect complex patterns
            patterns = self.pattern_matcher.detect_patterns(processed_event)
            processed_event.pattern_matches = patterns
            
            # Route to handlers
            event_type = processed_event.original_event.event_type
            route = self.event_routes.get(event_type)
            
            if route:
                # Execute registered handlers
                for handler in route.handlers:
                    try:
                        handler_start = datetime.now()
                        result = handler(processed_event)
                        handler_time = (datetime.now() - handler_start).total_seconds() * 1000
                        
                        processed_event.handler_results[handler.__name__] = {
                            'result': result,
                            'processing_time_ms': handler_time
                        }
                        
                        # Check handler timeout
                        if handler_time > route.max_processing_time_ms:
                            logger.warning("Handler exceeded time limit",
                                         handler=handler.__name__,
                                         time_ms=handler_time,
                                         limit_ms=route.max_processing_time_ms)
                        
                    except Exception as e:
                        logger.error("Handler error",
                                   handler=handler.__name__,
                                   event_type=event_type.value,
                                   error=str(e))
                        processed_event.handler_results[handler.__name__] = {
                            'error': str(e)
                        }
            else:
                # Execute default handlers
                for handler in self.default_handlers:
                    try:
                        handler(processed_event)
                    except Exception as e:
                        logger.error("Default handler error", error=str(e))
            
            # Mark processing complete
            processed_event.processing_end = datetime.now()
            processing_time = (processed_event.processing_end - start_time).total_seconds() * 1000
            
            # Track performance
            self.processing_times.append(processing_time)
            
            # Add to processed history
            self.processed_events.append(processed_event)
            
            # Performance monitoring
            if processing_time > self.latency_target_ms:
                self.latency_violations += 1
            
        except Exception as e:
            logger.error("Error processing event",
                        event_id=processed_event.metadata.event_id,
                        event_type=processed_event.original_event.event_type.value,
                        error=str(e))
            self.error_count += 1
    
    def _throughput_monitor_loop(self):
        """Monitor throughput and performance"""
        last_count = 0
        
        while self.running:
            time.sleep(1.0)  # Check every second
            
            # Calculate throughput
            current_count = self.event_count
            throughput = current_count - last_count
            self.throughput_history.append(throughput)
            last_count = current_count
            
            # Log performance metrics periodically
            if len(self.throughput_history) % 10 == 0:  # Every 10 seconds
                avg_throughput = np.mean(self.throughput_history)
                avg_latency = np.mean(self.processing_times) if self.processing_times else 0
                
                logger.debug("Event processing performance",
                           throughput_eps=avg_throughput,
                           avg_latency_ms=avg_latency,
                           queue_size=self.event_queue.qsize(),
                           latency_violations=self.latency_violations)
    
    def _infer_event_priority(self, event_type: EventType) -> EventPriority:
        """Infer event priority from event type"""
        if event_type in [EventType.EMERGENCY_STOP, EventType.SYSTEM_ERROR]:
            return EventPriority.EMERGENCY
        elif event_type in [EventType.RISK_BREACH, EventType.CONNECTION_LOST]:
            return EventPriority.CRITICAL
        elif event_type in [EventType.TRADE_QUALIFIED, EventType.STRATEGIC_DECISION]:
            return EventPriority.HIGH
        elif event_type in [EventType.VAR_UPDATE, EventType.POSITION_UPDATE]:
            return EventPriority.MEDIUM
        else:
            return EventPriority.LOW
    
    def _infer_event_category(self, event_type: EventType) -> EventCategory:
        """Infer event category from event type"""
        if event_type in [EventType.NEW_TICK, EventType.NEW_BAR, EventType.NEW_5MIN_BAR]:
            return EventCategory.MARKET_DATA
        elif event_type in [EventType.TRADE_QUALIFIED, EventType.EXECUTE_TRADE]:
            return EventCategory.TRADING_SIGNAL
        elif event_type in [EventType.RISK_BREACH, EventType.VAR_UPDATE]:
            return EventCategory.RISK_ALERT
        elif event_type in [EventType.SYSTEM_ERROR, EventType.CONNECTION_LOST]:
            return EventCategory.SYSTEM_HEALTH
        elif event_type == EventType.EMERGENCY_STOP:
            return EventCategory.EMERGENCY
        else:
            return EventCategory.COORDINATION
    
    def replay_events(self, 
                     start_time: datetime,
                     end_time: datetime,
                     event_types: Optional[List[EventType]] = None) -> List[ProcessedEvent]:
        """Replay events for debugging and analysis"""
        
        matching_events = []
        
        for event in self.event_history:
            # Check time range
            if not (start_time <= event.processing_start <= end_time):
                continue
            
            # Check event types filter
            if event_types and event.original_event.event_type not in event_types:
                continue
            
            matching_events.append(event)
        
        # Sort by processing time
        matching_events.sort(key=lambda e: e.processing_start)
        
        logger.info("Event replay completed",
                   event_count=len(matching_events),
                   start_time=start_time,
                   end_time=end_time)
        
        return matching_events
    
    def get_correlation_analysis(self, 
                               correlation_id: str) -> List[ProcessedEvent]:
        """Get all events with specific correlation ID"""
        
        correlated_events = []
        
        for event in self.processed_events:
            if event.metadata.correlation_id == correlation_id:
                correlated_events.append(event)
        
        # Sort by sequence number
        correlated_events.sort(key=lambda e: e.metadata.sequence_number)
        
        return correlated_events
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        # Calculate throughput metrics
        current_throughput = np.mean(self.throughput_history) if self.throughput_history else 0
        max_throughput = np.max(self.throughput_history) if self.throughput_history else 0
        
        # Calculate latency metrics
        avg_latency = np.mean(self.processing_times) if self.processing_times else 0
        p95_latency = np.percentile(self.processing_times, 95) if self.processing_times else 0
        p99_latency = np.percentile(self.processing_times, 99) if self.processing_times else 0
        
        # Calculate quality metrics
        total_events = max(1, self.event_count)
        error_rate = self.error_count / total_events
        latency_violation_rate = self.latency_violations / total_events
        
        return {
            'event_count': self.event_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'latency_violations': self.latency_violations,
            'latency_violation_rate': latency_violation_rate,
            'current_throughput_eps': current_throughput,
            'max_throughput_eps': max_throughput,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'meets_latency_target': avg_latency <= self.latency_target_ms,
            'queue_size': self.event_queue.qsize(),
            'processing_threads': len(self.processing_threads),
            'event_routes_count': len(self.event_routes),
            'pattern_rules_count': len(self.pattern_matcher.pattern_rules)
        }
    
    def add_default_handler(self, handler: Callable[[ProcessedEvent], None]):
        """Add default handler for unrouted events"""
        self.default_handlers.append(handler)
    
    def clear_history(self):
        """Clear event history to free memory"""
        self.event_history.clear()
        self.processed_events.clear()
        self.processing_times.clear()
        logger.info("Event history cleared")
    
    def emergency_clear_queue(self) -> int:
        """Emergency clear of event queue"""
        cleared_count = 0
        
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        logger.warning("Emergency queue clear", cleared_events=cleared_count)
        return cleared_count