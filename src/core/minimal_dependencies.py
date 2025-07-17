"""
Minimal dependencies module for backwards compatibility.

This module provides a minimal set of components that can be used 
when full AlgoSpace dependencies are not available.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Import required classes from events module
from .events import EventBus, Event, EventType, BarData


class MinimalComponentBase(ABC):
    """
    Minimal base class for components when full kernel is not available.
    
    This provides a simplified version of ComponentBase that doesn't 
    require the full AlgoSpace kernel infrastructure.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the minimal component.

        Args:
            name: Component name for identification
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component gracefully."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current component status."""
        pass

    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized


# Alias for backwards compatibility
ComponentBase = MinimalComponentBase