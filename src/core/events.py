from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Callable
import structlog
import threading

logger = structlog.get_logger()


class EventType(Enum):
    """Event types used throughout the system"""

    # Data Events
    NEW_TICK = "NEW_TICK"
    NEW_5MIN_BAR = "NEW_5MIN_BAR"
    NEW_30MIN_BAR = "NEW_30MIN_BAR"
    NEW_BAR = "NEW_BAR"  # Generic bar event (for backward compatibility)

    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    BACKTEST_COMPLETE = "BACKTEST_COMPLETE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    COMPONENT_STARTED = "COMPONENT_STARTED"
    COMPONENT_STOPPED = "COMPONENT_STOPPED"

    # Connection Events
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"

    # Indicator Events
    INDICATOR_UPDATE = "INDICATOR_UPDATE"
    INDICATORS_READY = "INDICATORS_READY"
    SYNERGY_DETECTED = "SYNERGY_DETECTED"

    # MARL Events
    TRADE_QUALIFIED = "TRADE_QUALIFIED"
    TRADE_REJECTED = "TRADE_REJECTED"
    EXECUTE_TRADE = "EXECUTE_TRADE"
    STRATEGIC_DECISION = "STRATEGIC_DECISION"
    TACTICAL_DECISION = "TACTICAL_DECISION"

    # Execution Events
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    TRADE_CLOSED = "TRADE_CLOSED"
    SHUTDOWN_REQUEST = "SHUTDOWN_REQUEST"

    # Risk Events
    RISK_BREACH = "RISK_BREACH"
    RISK_UPDATE = "RISK_UPDATE"
    VAR_UPDATE = "VAR_UPDATE"
    KELLY_SIZING = "KELLY_SIZING"
    POSITION_UPDATE = "POSITION_UPDATE"
    MARKET_STRESS = "MARKET_STRESS"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    POSITION_SIZE_UPDATE = "POSITION_SIZE_UPDATE"
    CORRELATION_SHOCK = "CORRELATION_SHOCK"
    
    # Crisis Intelligence Events
    CRISIS_PREMONITION_DETECTED = "CRISIS_PREMONITION_DETECTED"
    CRISIS_PATTERN_MATCH = "CRISIS_PATTERN_MATCH"
    EMERGENCY_PROTOCOL_ACTIVATED = "EMERGENCY_PROTOCOL_ACTIVATED"
    
    # XAI Pipeline Events
    XAI_DECISION_CAPTURED = "XAI_DECISION_CAPTURED"
    XAI_CONTEXT_PROCESSED = "XAI_CONTEXT_PROCESSED"
    XAI_EXPLANATION_GENERATED = "XAI_EXPLANATION_GENERATED"
    XAI_EXPLANATION_STREAMED = "XAI_EXPLANATION_STREAMED"
    XAI_CLIENT_CONNECTED = "XAI_CLIENT_CONNECTED"
    XAI_CLIENT_DISCONNECTED = "XAI_CLIENT_DISCONNECTED"
    XAI_PIPELINE_ERROR = "XAI_PIPELINE_ERROR"
    
    # Data Lineage Events
    DATA_LINEAGE_NODE_CREATED = "DATA_LINEAGE_NODE_CREATED"
    DATA_LINEAGE_EDGE_CREATED = "DATA_LINEAGE_EDGE_CREATED"
    DATA_LINEAGE_TRACE_BUILT = "DATA_LINEAGE_TRACE_BUILT"
    DATA_TRANSFORMATION_STARTED = "DATA_TRANSFORMATION_STARTED"
    DATA_TRANSFORMATION_COMPLETED = "DATA_TRANSFORMATION_COMPLETED"
    DATA_TRANSFORMATION_FAILED = "DATA_TRANSFORMATION_FAILED"
    DATA_QUALITY_DEGRADED = "DATA_QUALITY_DEGRADED"
    DATA_QUALITY_IMPROVED = "DATA_QUALITY_IMPROVED"
    DATA_LINEAGE_INTEGRITY_CHECK = "DATA_LINEAGE_INTEGRITY_CHECK"
    DATA_LINEAGE_CLEANUP = "DATA_LINEAGE_CLEANUP"
    
    # Data Validation Events
    DATA_VALIDATION_STARTED = "DATA_VALIDATION_STARTED"
    DATA_VALIDATION_COMPLETED = "DATA_VALIDATION_COMPLETED"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    DATA_SCHEMA_VALIDATION_ERROR = "DATA_SCHEMA_VALIDATION_ERROR"
    DATA_QUALITY_CHECK_PASSED = "DATA_QUALITY_CHECK_PASSED"
    DATA_QUALITY_CHECK_FAILED = "DATA_QUALITY_CHECK_FAILED"
    DATA_QUALITY_REPORT_GENERATED = "DATA_QUALITY_REPORT_GENERATED"
    
    # ML Anomaly Detection Events
    ML_ANOMALY_DETECTED = "ML_ANOMALY_DETECTED"
    ML_MODEL_TRAINED = "ML_MODEL_TRAINED"
    ML_MODEL_RETRAINED = "ML_MODEL_RETRAINED"
    ML_MODEL_PERFORMANCE_DEGRADED = "ML_MODEL_PERFORMANCE_DEGRADED"
    STATISTICAL_ANOMALY_DETECTED = "STATISTICAL_ANOMALY_DETECTED"
    PATTERN_ANOMALY_DETECTED = "PATTERN_ANOMALY_DETECTED"
    ENSEMBLE_ANOMALY_CONFIRMED = "ENSEMBLE_ANOMALY_CONFIRMED"


@dataclass
class TickData:
    """Standard tick data structure"""

    symbol: str
    timestamp: datetime
    price: float
    volume: int


@dataclass
class BarData:
    """Standard bar data structure"""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: int  # 5 or 30 (minutes)


@dataclass
class Event:
    """Base event structure"""

    event_type: EventType
    timestamp: datetime
    payload: Any
    source: str


class EventBus:
    """Central event bus for system-wide communication"""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._logger = structlog.get_logger(self.__class__.__name__)
        self._lock = threading.RLock()  # Use RLock for nested locking

    def subscribe(
        self, event_type: EventType, callback: Callable[[Event], None]
    ) -> None:
        """Subscribe to an event type"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            self._logger.debug("Subscriber registered", event_type=event_type.value)

    def unsubscribe(
        self, event_type: EventType, callback: Callable[[Event], None]
    ) -> None:
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    self._logger.debug("Subscriber removed", event_type=event_type.value)
                except ValueError:
                    self._logger.warning(
                        "Callback not found for unsubscribe", event_type=event_type.value
                    )

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        # Make a copy of subscribers to avoid holding lock during callbacks
        callbacks = []
        with self._lock:
            if event.event_type in self._subscribers:
                callbacks = self._subscribers[event.event_type].copy()
        
        # Execute callbacks outside of lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(
                    "Error in event callback",
                    event_type=event.event_type.value,
                    error=str(e),
                )

        # Log high-frequency events at debug level only
        if event.event_type in [EventType.NEW_TICK, EventType.NEW_BAR]:
            self._logger.debug("Event published", event_type=event.event_type.value)
        else:
            self._logger.info("Event published", event_type=event.event_type.value)

    def create_event(self, event_type: EventType, payload: Any, source: str) -> Event:
        """Factory method to create events with timestamp"""
        return Event(
            event_type=event_type,
            timestamp=datetime.now(),
            payload=payload,
            source=source,
        )

    def dispatch_forever(self) -> None:
        """Dispatch events forever (mock implementation for testing)"""
        import time

        self._logger.info("Event bus dispatch started")
        try:
            while True:
                time.sleep(1)  # Basic event loop
        except KeyboardInterrupt:
            self._logger.info("Event bus dispatch interrupted")

    def stop(self) -> None:
        """Stop the event bus"""
        self._logger.info("Event bus stopped")
