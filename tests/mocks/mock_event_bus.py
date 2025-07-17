"""
Mock Event Bus for testing

Provides a testable event bus that tracks all published events
and allows controlled event injection.
"""

from typing import List, Dict, Any, Callable
from src.core.minimal_dependencies import MinimalEventBus, Event, EventType


class MockEventBus(MinimalEventBus):
    """
    Mock event bus for testing
    
    Tracks all published events and provides test utilities.
    """
    
    def __init__(self):
        super().__init__()
        self.published_events: List[Event] = []
        self.event_history: Dict[EventType, List[Event]] = {}
        self._enabled = True
        
    def publish(self, event: Event) -> None:
        """
        Override to track published events
        
        Args:
            event: Event to publish
        """
        if self._enabled:
            # Track event
            self.published_events.append(event)
            
            # Track by type
            if event.event_type not in self.event_history:
                self.event_history[event.event_type] = []
            self.event_history[event.event_type].append(event)
            
            # Call parent to actually publish
            super().publish(event)
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """
        Get all events of a specific type
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of events of that type
        """
        return self.event_history.get(event_type, [])
    
    def get_last_event(self, event_type: EventType = None) -> Event:
        """
        Get the last published event
        
        Args:
            event_type: Optional type filter
            
        Returns:
            Last event matching criteria or None
        """
        if event_type:
            events = self.get_events_by_type(event_type)
            return events[-1] if events else None
        else:
            return self.published_events[-1] if self.published_events else None
    
    def clear_history(self):
        """Clear all tracked events"""
        self.published_events.clear()
        self.event_history.clear()
    
    def disable(self):
        """Disable event publishing (for testing)"""
        self._enabled = False
        
    def enable(self):
        """Re-enable event publishing"""
        self._enabled = True
    
    def assert_event_published(self, event_type: EventType, count: int = None) -> bool:
        """
        Assert that an event type was published
        
        Args:
            event_type: Type of event to check
            count: Optional expected count
            
        Returns:
            True if assertion passes
            
        Raises:
            AssertionError: If event not published or count mismatch
        """
        events = self.get_events_by_type(event_type)
        
        if not events:
            raise AssertionError(f"No events of type {event_type} were published")
            
        if count is not None and len(events) != count:
            raise AssertionError(
                f"Expected {count} events of type {event_type}, "
                f"but got {len(events)}"
            )
            
        return True
    
    def wait_for_event(self, event_type: EventType, timeout: float = 1.0) -> Event:
        """
        Wait for a specific event type (simplified for testing)
        
        Args:
            event_type: Type of event to wait for
            timeout: Timeout in seconds
            
        Returns:
            The event if found, None if timeout
        """
        # In real implementation, this would use threading
        # For testing, we just check if event exists
        events = self.get_events_by_type(event_type)
        return events[-1] if events else None