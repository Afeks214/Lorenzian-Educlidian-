"""
Trading System Controller - Master Switch for Risk Management Integration

This controller provides a centralized master switch system that integrates with
the kill switch mechanism to control all risk management components.

Key Features:
- Thread-safe ON/OFF state management
- Integration with kill switch system
- Cached state preservation during OFF periods
- Comprehensive logging and monitoring
- Graceful state transitions
"""

import threading
import time
import logging
import json
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager

# Import the kill switch system
from src.safety.kill_switch import get_kill_switch, TradingSystemKillSwitch

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operational states."""
    ON = "on"
    OFF = "off"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class StateTransition:
    """Record of state transitions."""
    timestamp: datetime
    from_state: SystemState
    to_state: SystemState
    reason: str
    initiator: str
    metadata: Dict[str, Any]

class TradingSystemController:
    """
    Master switch controller for trading system components.
    
    This controller provides centralized state management for all risk management
    components, ensuring thread-safe operations and graceful state transitions.
    """
    
    def __init__(self, enable_kill_switch_integration: bool = True):
        """Initialize the trading system controller."""
        self._state = SystemState.OFF
        self._previous_state = SystemState.OFF
        self._state_lock = threading.RLock()
        self._state_change_callbacks = []
        self._component_states = {}
        self._cached_values = {}
        self._transition_history = []
        self._kill_switch_integration = enable_kill_switch_integration
        
        # Performance tracking
        self._state_check_count = 0
        self._last_state_check = time.time()
        
        # Initialize kill switch integration
        if self._kill_switch_integration:
            self._setup_kill_switch_integration()
        
        logger.info(f"Trading System Controller initialized (kill_switch_integration: {enable_kill_switch_integration})")
    
    def _setup_kill_switch_integration(self):
        """Setup integration with kill switch system."""
        try:
            kill_switch = get_kill_switch()
            if kill_switch:
                # Monitor kill switch state
                def check_kill_switch():
                    """Check kill switch state and update system accordingly."""
                    if kill_switch.is_active():
                        logger.warning("Kill switch activated - stopping trading system")
                        self.stop_system(
                            reason="kill_switch_activated",
                            initiator="kill_switch_monitor"
                        )
                
                # Register periodic check (would be called by monitoring thread)
                self._kill_switch_checker = check_kill_switch
                logger.info("Kill switch integration enabled")
            else:
                logger.warning("Kill switch not available - running without integration")
                self._kill_switch_integration = False
        except Exception as e:
            logger.error(f"Failed to setup kill switch integration: {e}")
            self._kill_switch_integration = False
    
    def start_system(self, reason: str = "manual_start", initiator: str = "operator") -> bool:
        """
        Start the trading system.
        
        Args:
            reason: Reason for starting the system
            initiator: Who initiated the start
            
        Returns:
            True if system started successfully
        """
        with self._state_lock:
            if self._state == SystemState.ON:
                logger.warning("System already running")
                return True
            
            if self._state == SystemState.STARTING:
                logger.warning("System already starting")
                return True
            
            logger.info(f"Starting trading system: {reason}")
            
            # Transition to starting state
            self._transition_state(
                SystemState.STARTING,
                reason=reason,
                initiator=initiator
            )
            
            try:
                # Notify components of system start
                self._notify_state_change(SystemState.STARTING)
                
                # Simulate startup process
                time.sleep(0.1)  # Brief startup delay
                
                # Transition to ON state
                self._transition_state(
                    SystemState.ON,
                    reason="startup_complete",
                    initiator=initiator
                )
                
                # Notify components
                self._notify_state_change(SystemState.ON)
                
                logger.info("✅ Trading system started successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start system: {e}")
                self._transition_state(
                    SystemState.ERROR,
                    reason=f"startup_failed: {str(e)}",
                    initiator=initiator
                )
                return False
    
    def stop_system(self, reason: str = "manual_stop", initiator: str = "operator") -> bool:
        """
        Stop the trading system.
        
        Args:
            reason: Reason for stopping the system
            initiator: Who initiated the stop
            
        Returns:
            True if system stopped successfully
        """
        with self._state_lock:
            if self._state == SystemState.OFF:
                logger.warning("System already stopped")
                return True
            
            if self._state == SystemState.STOPPING:
                logger.warning("System already stopping")
                return True
            
            logger.info(f"Stopping trading system: {reason}")
            
            # Transition to stopping state
            self._transition_state(
                SystemState.STOPPING,
                reason=reason,
                initiator=initiator
            )
            
            try:
                # Notify components of system stop
                self._notify_state_change(SystemState.STOPPING)
                
                # Simulate shutdown process
                time.sleep(0.1)  # Brief shutdown delay
                
                # Transition to OFF state
                self._transition_state(
                    SystemState.OFF,
                    reason="shutdown_complete",
                    initiator=initiator
                )
                
                # Notify components
                self._notify_state_change(SystemState.OFF)
                
                logger.info("🛑 Trading system stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop system: {e}")
                self._transition_state(
                    SystemState.ERROR,
                    reason=f"shutdown_failed: {str(e)}",
                    initiator=initiator
                )
                return False
    
    def _transition_state(self, new_state: SystemState, reason: str, initiator: str):
        """Transition to new state and record the transition."""
        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        
        # Record transition
        transition = StateTransition(
            timestamp=datetime.now(),
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            initiator=initiator,
            metadata={
                "component_states": self._component_states.copy(),
                "cached_values_count": len(self._cached_values)
            }
        )
        
        self._transition_history.append(transition)
        
        # Keep only recent transitions
        if len(self._transition_history) > 100:
            self._transition_history = self._transition_history[-100:]
        
        logger.info(f"State transition: {old_state.value} -> {new_state.value} ({reason})")
    
    def _notify_state_change(self, new_state: SystemState):
        """Notify registered callbacks of state changes."""
        for callback in self._state_change_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")
    
    def is_system_on(self) -> bool:
        """
        Check if the system is ON.
        
        This is the main method used by risk management components
        to check if they should process new calculations.
        
        Returns:
            True if system is ON and operational
        """
        with self._state_lock:
            self._state_check_count += 1
            self._last_state_check = time.time()
            
            # Check kill switch if enabled
            if self._kill_switch_integration and hasattr(self, '_kill_switch_checker'):
                try:
                    self._kill_switch_checker()
                except Exception as e:
                    logger.error(f"Kill switch check failed: {e}")
            
            return self._state == SystemState.ON
    
    def is_system_off(self) -> bool:
        """Check if the system is OFF."""
        with self._state_lock:
            return self._state == SystemState.OFF
    
    def get_system_state(self) -> SystemState:
        """Get current system state."""
        with self._state_lock:
            return self._state
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._state_lock:
            return {
                "state": self._state.value,
                "previous_state": self._previous_state.value,
                "uptime_seconds": time.time() - self._last_state_check if self._state == SystemState.ON else 0,
                "state_check_count": self._state_check_count,
                "last_state_check": self._last_state_check,
                "component_states": self._component_states.copy(),
                "cached_values_count": len(self._cached_values),
                "kill_switch_integration": self._kill_switch_integration,
                "transition_count": len(self._transition_history),
                "last_transition": asdict(self._transition_history[-1]) if self._transition_history else None
            }
    
    def register_state_change_callback(self, callback: Callable[[SystemState], None]):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def register_component(self, component_name: str, component_info: Dict[str, Any]):
        """Register a component with the controller."""
        with self._state_lock:
            self._component_states[component_name] = {
                "info": component_info,
                "registered_at": datetime.now(),
                "last_update": datetime.now()
            }
            logger.info(f"Component registered: {component_name}")
    
    def update_component_status(self, component_name: str, status: Dict[str, Any]):
        """Update component status."""
        with self._state_lock:
            if component_name in self._component_states:
                self._component_states[component_name]["status"] = status
                self._component_states[component_name]["last_update"] = datetime.now()
    
    def cache_value(self, key: str, value: Any, ttl_seconds: int = 300):
        """
        Cache a value for use when system is OFF.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
        """
        with self._state_lock:
            self._cached_values[key] = {
                "value": value,
                "cached_at": time.time(),
                "ttl_seconds": ttl_seconds
            }
    
    def get_cached_value(self, key: str) -> Optional[Any]:
        """
        Get cached value, checking TTL.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if valid, None otherwise
        """
        with self._state_lock:
            if key not in self._cached_values:
                return None
            
            cached_item = self._cached_values[key]
            current_time = time.time()
            
            # Check TTL
            if current_time - cached_item["cached_at"] > cached_item["ttl_seconds"]:
                # Expired, remove and return None
                del self._cached_values[key]
                return None
            
            return cached_item["value"]
    
    def clear_cache(self):
        """Clear all cached values."""
        with self._state_lock:
            self._cached_values.clear()
            logger.info("Cache cleared")
    
    @contextmanager
    def require_system_on(self, operation_name: str = "operation"):
        """
        Context manager that requires system to be ON.
        
        Args:
            operation_name: Name of the operation for logging
            
        Raises:
            RuntimeError: If system is not ON
        """
        if not self.is_system_on():
            raise RuntimeError(f"Cannot perform {operation_name}: system is {self._state.value}")
        
        try:
            yield
        finally:
            pass
    
    def get_transition_history(self, limit: int = 10) -> list:
        """Get recent state transitions."""
        with self._state_lock:
            return [asdict(t) for t in self._transition_history[-limit:]]
    
    def emergency_stop(self, reason: str = "emergency"):
        """Emergency stop via kill switch integration."""
        if self._kill_switch_integration:
            kill_switch = get_kill_switch()
            if kill_switch:
                kill_switch.emergency_stop(reason, human_override=True)
            else:
                logger.error("Kill switch not available for emergency stop")
                self.stop_system(reason="emergency_stop_fallback", initiator="emergency")
        else:
            self.stop_system(reason="emergency_stop", initiator="emergency")
    
    def reset_system(self, reason: str = "reset"):
        """Reset system to clean state."""
        with self._state_lock:
            logger.info(f"Resetting system: {reason}")
            
            # Stop system first
            self.stop_system(reason=f"reset: {reason}", initiator="reset")
            
            # Clear cached values
            self.clear_cache()
            
            # Reset counters
            self._state_check_count = 0
            self._last_state_check = time.time()
            
            # Keep only recent transition history
            if len(self._transition_history) > 10:
                self._transition_history = self._transition_history[-10:]
            
            logger.info("System reset complete")


# Global controller instance
_global_controller = None

def initialize_controller(enable_kill_switch_integration: bool = True) -> TradingSystemController:
    """Initialize global trading system controller."""
    global _global_controller
    _global_controller = TradingSystemController(enable_kill_switch_integration)
    return _global_controller

def get_controller() -> Optional[TradingSystemController]:
    """Get global trading system controller."""
    return _global_controller

def is_system_on() -> bool:
    """Global function to check if system is ON."""
    controller = get_controller()
    if controller:
        return controller.is_system_on()
    else:
        logger.warning("Trading system controller not initialized - assuming OFF")
        return False

def get_system_status() -> Dict[str, Any]:
    """Global function to get system status."""
    controller = get_controller()
    if controller:
        return controller.get_system_status()
    else:
        return {
            "state": "unknown",
            "error": "controller_not_initialized"
        }

if __name__ == "__main__":
    # Demo usage
    print("Testing Trading System Controller...")
    
    # Initialize controller
    controller = initialize_controller()
    
    # Test basic operations
    print(f"Initial state: {controller.get_system_state()}")
    
    # Start system
    controller.start_system("test_start", "demo")
    print(f"After start: {controller.get_system_state()}")
    
    # Test system checks
    print(f"Is system on: {controller.is_system_on()}")
    print(f"Is system off: {controller.is_system_off()}")
    
    # Test caching
    controller.cache_value("test_var", 123.456)
    print(f"Cached value: {controller.get_cached_value('test_var')}")
    
    # Stop system
    controller.stop_system("test_stop", "demo")
    print(f"After stop: {controller.get_system_state()}")
    
    # Test cached value still available
    print(f"Cached value after stop: {controller.get_cached_value('test_var')}")
    
    # Get status
    status = controller.get_system_status()
    print(f"System status: {json.dumps(status, indent=2, default=str)}")