"""
Trading System Controller - Master Switch Implementation

This module provides centralized control for the entire trading system,
including emergency shutdown, state management, and component coordination.
"""

import threading
import time
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import weakref

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Trading system states."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    EMERGENCY_STOPPED = "emergency_stopped"
    ERROR = "error"

class ComponentStatus(Enum):
    """Component status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class StateTransition:
    """State transition record."""
    id: str
    from_state: SystemState
    to_state: SystemState
    timestamp: datetime
    initiator: str
    reason: str
    success: bool
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ComponentInfo:
    """Component information."""
    name: str
    status: ComponentStatus
    last_heartbeat: datetime
    health_check_interval: float
    metadata: Dict[str, Any]

class TradingSystemController:
    """
    Master switch for the trading system.
    
    Provides centralized control for system lifecycle, emergency shutdown,
    and component coordination with high reliability and safety.
    """
    
    def __init__(self, 
                 max_concurrent_operations: int = 10,
                 heartbeat_timeout: float = 30.0,
                 state_persistence_path: str = "/tmp/trading_system_state.json"):
        """Initialize the trading system controller."""
        
        # System state
        self._state = SystemState.INACTIVE
        self._state_lock = threading.RLock()
        self._state_history: List[StateTransition] = []
        self._state_persistence_path = state_persistence_path
        
        # Component management
        self._components: Dict[str, ComponentInfo] = {}
        self._component_lock = threading.RLock()
        self._heartbeat_timeout = heartbeat_timeout
        
        # Operation management
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_operations)
        self._active_operations: Dict[str, Future] = {}
        self._operation_lock = threading.Lock()
        
        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._event_lock = threading.Lock()
        
        # Monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._performance_metrics: Dict[str, Any] = {}
        
        # Safety mechanisms
        self._safety_checks: List[Callable[[], bool]] = []
        self._emergency_callbacks: List[Callable] = []
        self._failsafe_active = False
        
        # Persistence
        self._auto_save_enabled = True
        self._last_save_time = time.time()
        
        logger.info("Trading system controller initialized")
    
    def start_system(self, timeout: float = 30.0, force: bool = False) -> bool:
        """
        Start the trading system.
        
        Args:
            timeout: Maximum time to wait for startup
            force: Force start even if safety checks fail
            
        Returns:
            True if system started successfully
        """
        with self._state_lock:
            if self._state == SystemState.ACTIVE:
                logger.warning("System already active")
                return True
            
            if self._state not in [SystemState.INACTIVE, SystemState.PAUSED]:
                logger.error(f"Cannot start system from state: {self._state}")
                return False
            
            # Record state transition
            transition_id = str(uuid.uuid4())
            old_state = self._state
            self._state = SystemState.STARTING
            
            transition = StateTransition(
                id=transition_id,
                from_state=old_state,
                to_state=SystemState.STARTING,
                timestamp=datetime.now(timezone.utc),
                initiator="system_controller",
                reason="manual_start",
                success=False
            )
            
            start_time = time.time()
            
            try:
                # Run safety checks
                if not force and not self._run_safety_checks():
                    logger.error("Safety checks failed, cannot start system")
                    self._state = old_state
                    transition.success = False
                    transition.to_state = old_state
                    self._state_history.append(transition)
                    return False
                
                # Start components
                if not self._start_components(timeout):
                    logger.error("Component startup failed")
                    self._state = SystemState.ERROR
                    transition.success = False
                    transition.to_state = SystemState.ERROR
                    self._state_history.append(transition)
                    return False
                
                # Start monitoring
                self._start_monitoring()
                
                # Update state
                self._state = SystemState.ACTIVE
                transition.success = True
                transition.to_state = SystemState.ACTIVE
                transition.duration_ms = (time.time() - start_time) * 1000
                
                self._state_history.append(transition)
                self._emit_event("system_started", {"transition": asdict(transition)})
                
                # Persist state
                self._persist_state()
                
                logger.info("Trading system started successfully")
                return True
                
            except Exception as e:
                logger.error(f"System startup failed: {e}")
                self._state = SystemState.ERROR
                transition.success = False
                transition.to_state = SystemState.ERROR
                transition.metadata = {"error": str(e)}
                self._state_history.append(transition)
                return False
    
    def stop_system(self, timeout: float = 30.0, force: bool = False) -> bool:
        """
        Stop the trading system gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
            force: Force shutdown even if operations are pending
            
        Returns:
            True if system stopped successfully
        """
        with self._state_lock:
            if self._state == SystemState.INACTIVE:
                logger.warning("System already inactive")
                return True
            
            if self._state == SystemState.STOPPING:
                logger.warning("System already stopping")
                return False
            
            # Record state transition
            transition_id = str(uuid.uuid4())
            old_state = self._state
            self._state = SystemState.STOPPING
            
            transition = StateTransition(
                id=transition_id,
                from_state=old_state,
                to_state=SystemState.STOPPING,
                timestamp=datetime.now(timezone.utc),
                initiator="system_controller",
                reason="manual_stop",
                success=False
            )
            
            start_time = time.time()
            
            try:
                # Wait for active operations to complete
                if not force:
                    self._wait_for_operations(timeout / 2)
                
                # Stop monitoring
                self._stop_monitoring()
                
                # Stop components
                self._stop_components(timeout / 2)
                
                # Cleanup
                self._cleanup_resources()
                
                # Update state
                self._state = SystemState.INACTIVE
                transition.success = True
                transition.to_state = SystemState.INACTIVE
                transition.duration_ms = (time.time() - start_time) * 1000
                
                self._state_history.append(transition)
                self._emit_event("system_stopped", {"transition": asdict(transition)})
                
                # Persist state
                self._persist_state()
                
                logger.info("Trading system stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"System shutdown failed: {e}")
                self._state = SystemState.ERROR
                transition.success = False
                transition.to_state = SystemState.ERROR
                transition.metadata = {"error": str(e)}
                self._state_history.append(transition)
                return False
    
    def emergency_stop(self, reason: str = "manual", initiator: str = "operator") -> bool:
        """
        Emergency stop the system immediately.
        
        Args:
            reason: Reason for emergency stop
            initiator: Who initiated the emergency stop
            
        Returns:
            True if emergency stop successful
        """
        with self._state_lock:
            # Record state transition
            transition_id = str(uuid.uuid4())
            old_state = self._state
            self._state = SystemState.EMERGENCY_STOPPED
            
            transition = StateTransition(
                id=transition_id,
                from_state=old_state,
                to_state=SystemState.EMERGENCY_STOPPED,
                timestamp=datetime.now(timezone.utc),
                initiator=initiator,
                reason=f"emergency_stop: {reason}",
                success=False
            )
            
            start_time = time.time()
            
            try:
                logger.critical(f"EMERGENCY STOP initiated by {initiator}: {reason}")
                
                # Cancel all operations immediately
                self._cancel_all_operations()
                
                # Emergency stop components
                self._emergency_stop_components()
                
                # Execute emergency callbacks
                self._execute_emergency_callbacks()
                
                # Stop monitoring
                self._stop_monitoring()
                
                # Activate failsafe
                self._failsafe_active = True
                
                transition.success = True
                transition.duration_ms = (time.time() - start_time) * 1000
                
                self._state_history.append(transition)
                self._emit_event("emergency_stop", {"transition": asdict(transition)})
                
                # Persist state
                self._persist_state()
                
                logger.critical("Emergency stop completed successfully")
                return True
                
            except Exception as e:
                logger.critical(f"Emergency stop failed: {e}")
                transition.success = False
                transition.metadata = {"error": str(e)}
                self._state_history.append(transition)
                return False
    
    def pause_system(self, timeout: float = 10.0) -> bool:
        """
        Pause the trading system temporarily.
        
        Args:
            timeout: Maximum time to wait for pause
            
        Returns:
            True if system paused successfully
        """
        with self._state_lock:
            if self._state != SystemState.ACTIVE:
                logger.error(f"Cannot pause system from state: {self._state}")
                return False
            
            # Record state transition
            transition_id = str(uuid.uuid4())
            old_state = self._state
            self._state = SystemState.PAUSING
            
            transition = StateTransition(
                id=transition_id,
                from_state=old_state,
                to_state=SystemState.PAUSING,
                timestamp=datetime.now(timezone.utc),
                initiator="system_controller",
                reason="manual_pause",
                success=False
            )
            
            start_time = time.time()
            
            try:
                # Pause components
                self._pause_components(timeout)
                
                # Update state
                self._state = SystemState.PAUSED
                transition.success = True
                transition.to_state = SystemState.PAUSED
                transition.duration_ms = (time.time() - start_time) * 1000
                
                self._state_history.append(transition)
                self._emit_event("system_paused", {"transition": asdict(transition)})
                
                logger.info("Trading system paused successfully")
                return True
                
            except Exception as e:
                logger.error(f"System pause failed: {e}")
                self._state = SystemState.ERROR
                transition.success = False
                transition.to_state = SystemState.ERROR
                transition.metadata = {"error": str(e)}
                self._state_history.append(transition)
                return False
    
    def resume_system(self, timeout: float = 10.0) -> bool:
        """
        Resume the trading system from pause.
        
        Args:
            timeout: Maximum time to wait for resume
            
        Returns:
            True if system resumed successfully
        """
        with self._state_lock:
            if self._state != SystemState.PAUSED:
                logger.error(f"Cannot resume system from state: {self._state}")
                return False
            
            # Record state transition
            transition_id = str(uuid.uuid4())
            old_state = self._state
            self._state = SystemState.STARTING
            
            transition = StateTransition(
                id=transition_id,
                from_state=old_state,
                to_state=SystemState.STARTING,
                timestamp=datetime.now(timezone.utc),
                initiator="system_controller",
                reason="manual_resume",
                success=False
            )
            
            start_time = time.time()
            
            try:
                # Resume components
                self._resume_components(timeout)
                
                # Update state
                self._state = SystemState.ACTIVE
                transition.success = True
                transition.to_state = SystemState.ACTIVE
                transition.duration_ms = (time.time() - start_time) * 1000
                
                self._state_history.append(transition)
                self._emit_event("system_resumed", {"transition": asdict(transition)})
                
                logger.info("Trading system resumed successfully")
                return True
                
            except Exception as e:
                logger.error(f"System resume failed: {e}")
                self._state = SystemState.ERROR
                transition.success = False
                transition.to_state = SystemState.ERROR
                transition.metadata = {"error": str(e)}
                self._state_history.append(transition)
                return False
    
    def get_state(self) -> SystemState:
        """Get current system state."""
        with self._state_lock:
            return self._state
    
    def get_state_history(self, limit: int = 100) -> List[StateTransition]:
        """Get system state history."""
        with self._state_lock:
            return self._state_history[-limit:]
    
    def register_component(self, 
                         name: str, 
                         health_check_interval: float = 10.0,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a system component.
        
        Args:
            name: Component name
            health_check_interval: Health check interval in seconds
            metadata: Additional component metadata
            
        Returns:
            True if component registered successfully
        """
        with self._component_lock:
            if name in self._components:
                logger.warning(f"Component {name} already registered")
                return False
            
            component = ComponentInfo(
                name=name,
                status=ComponentStatus.UNKNOWN,
                last_heartbeat=datetime.now(timezone.utc),
                health_check_interval=health_check_interval,
                metadata=metadata or {}
            )
            
            self._components[name] = component
            logger.info(f"Component {name} registered")
            return True
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a system component.
        
        Args:
            name: Component name
            
        Returns:
            True if component unregistered successfully
        """
        with self._component_lock:
            if name not in self._components:
                logger.warning(f"Component {name} not registered")
                return False
            
            del self._components[name]
            logger.info(f"Component {name} unregistered")
            return True
    
    def update_component_status(self, name: str, status: ComponentStatus, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update component status.
        
        Args:
            name: Component name
            status: New status
            metadata: Additional metadata
            
        Returns:
            True if status updated successfully
        """
        with self._component_lock:
            if name not in self._components:
                logger.warning(f"Component {name} not registered")
                return False
            
            component = self._components[name]
            component.status = status
            component.last_heartbeat = datetime.now(timezone.utc)
            if metadata:
                component.metadata.update(metadata)
            
            logger.debug(f"Component {name} status updated to {status}")
            return True
    
    def get_component_status(self, name: str) -> Optional[ComponentInfo]:
        """Get component status."""
        with self._component_lock:
            return self._components.get(name)
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """Get all component statuses."""
        with self._component_lock:
            return self._components.copy()
    
    def add_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Add event handler.
        
        Args:
            event_type: Event type to listen for
            handler: Handler function
            
        Returns:
            True if handler added successfully
        """
        with self._event_lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            
            self._event_handlers[event_type].append(handler)
            logger.debug(f"Event handler added for {event_type}")
            return True
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Remove event handler.
        
        Args:
            event_type: Event type
            handler: Handler function to remove
            
        Returns:
            True if handler removed successfully
        """
        with self._event_lock:
            if event_type not in self._event_handlers:
                return False
            
            try:
                self._event_handlers[event_type].remove(handler)
                logger.debug(f"Event handler removed for {event_type}")
                return True
            except ValueError:
                return False
    
    def add_safety_check(self, check_func: Callable[[], bool]) -> bool:
        """
        Add safety check function.
        
        Args:
            check_func: Function that returns True if safe
            
        Returns:
            True if check added successfully
        """
        self._safety_checks.append(check_func)
        logger.debug("Safety check added")
        return True
    
    def add_emergency_callback(self, callback: Callable) -> bool:
        """
        Add emergency callback.
        
        Args:
            callback: Function to call on emergency stop
            
        Returns:
            True if callback added successfully
        """
        self._emergency_callbacks.append(callback)
        logger.debug("Emergency callback added")
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "state": self._state.value,
            "uptime": time.time() - self._last_save_time,
            "component_count": len(self._components),
            "active_operations": len(self._active_operations),
            "state_transitions": len(self._state_history),
            "failsafe_active": self._failsafe_active
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        with self._state_lock:
            if self._state not in [SystemState.ACTIVE, SystemState.PAUSED]:
                return False
            
            # Check component health
            with self._component_lock:
                for component in self._components.values():
                    if component.status == ComponentStatus.FAILED:
                        return False
                    
                    # Check heartbeat timeout
                    elapsed = (datetime.now(timezone.utc) - component.last_heartbeat).total_seconds()
                    if elapsed > component.health_check_interval * 2:
                        return False
            
            return True
    
    def _run_safety_checks(self) -> bool:
        """Run all safety checks."""
        for check in self._safety_checks:
            try:
                if not check():
                    logger.warning(f"Safety check failed: {check.__name__}")
                    return False
            except Exception as e:
                logger.error(f"Safety check error: {e}")
                return False
        
        logger.debug("All safety checks passed")
        return True
    
    def _start_components(self, timeout: float) -> bool:
        """Start all components."""
        # Simulate component startup
        time.sleep(0.1)  # Simulate startup time
        
        # Update component statuses
        with self._component_lock:
            for component in self._components.values():
                component.status = ComponentStatus.HEALTHY
                component.last_heartbeat = datetime.now(timezone.utc)
        
        return True
    
    def _stop_components(self, timeout: float) -> bool:
        """Stop all components."""
        # Simulate component shutdown
        time.sleep(0.1)  # Simulate shutdown time
        
        # Update component statuses
        with self._component_lock:
            for component in self._components.values():
                component.status = ComponentStatus.UNKNOWN
        
        return True
    
    def _emergency_stop_components(self):
        """Emergency stop all components."""
        # Immediate component shutdown
        with self._component_lock:
            for component in self._components.values():
                component.status = ComponentStatus.FAILED
        
        logger.info("All components emergency stopped")
    
    def _pause_components(self, timeout: float):
        """Pause all components."""
        # Simulate component pause
        time.sleep(0.05)
        
        with self._component_lock:
            for component in self._components.values():
                component.metadata["paused"] = True
    
    def _resume_components(self, timeout: float):
        """Resume all components."""
        # Simulate component resume
        time.sleep(0.05)
        
        with self._component_lock:
            for component in self._components.values():
                component.metadata.pop("paused", None)
    
    def _start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def _stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Check component health
                self._check_component_health()
                
                # Auto-save state
                if self._auto_save_enabled and time.time() - self._last_save_time > 60:
                    self._persist_state()
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5.0)
    
    def _check_component_health(self):
        """Check health of all components."""
        now = datetime.now(timezone.utc)
        
        with self._component_lock:
            for component in self._components.values():
                elapsed = (now - component.last_heartbeat).total_seconds()
                
                if elapsed > component.health_check_interval * 2:
                    if component.status != ComponentStatus.FAILED:
                        component.status = ComponentStatus.FAILED
                        logger.warning(f"Component {component.name} health check timeout")
                        
                        # Emit event
                        self._emit_event("component_health_timeout", {
                            "component": component.name,
                            "elapsed": elapsed,
                            "timeout": component.health_check_interval * 2
                        })
    
    def _wait_for_operations(self, timeout: float):
        """Wait for active operations to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._operation_lock:
                if not self._active_operations:
                    return
            
            time.sleep(0.1)
        
        logger.warning("Operations did not complete within timeout")
    
    def _cancel_all_operations(self):
        """Cancel all active operations."""
        with self._operation_lock:
            for operation_id, future in self._active_operations.items():
                future.cancel()
                logger.info(f"Cancelled operation {operation_id}")
            
            self._active_operations.clear()
    
    def _execute_emergency_callbacks(self):
        """Execute all emergency callbacks."""
        for callback in self._emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def _cleanup_resources(self):
        """Cleanup system resources."""
        # Cancel remaining operations
        self._cancel_all_operations()
        
        # Clear event handlers
        with self._event_lock:
            self._event_handlers.clear()
        
        logger.info("Resources cleaned up")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit system event."""
        with self._event_lock:
            handlers = self._event_handlers.get(event_type, [])
            
            for handler in handlers:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    def _persist_state(self):
        """Persist system state to disk."""
        try:
            state_data = {
                "state": self._state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    name: {
                        "status": comp.status.value,
                        "last_heartbeat": comp.last_heartbeat.isoformat(),
                        "health_check_interval": comp.health_check_interval,
                        "metadata": comp.metadata
                    }
                    for name, comp in self._components.items()
                },
                "failsafe_active": self._failsafe_active
            }
            
            with open(self._state_persistence_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self._last_save_time = time.time()
            logger.debug("System state persisted")
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    def load_state(self) -> bool:
        """Load system state from disk."""
        try:
            with open(self._state_persistence_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            self._state = SystemState(state_data["state"])
            self._failsafe_active = state_data.get("failsafe_active", False)
            
            # Restore components
            for name, comp_data in state_data.get("components", {}).items():
                component = ComponentInfo(
                    name=name,
                    status=ComponentStatus(comp_data["status"]),
                    last_heartbeat=datetime.fromisoformat(comp_data["last_heartbeat"]),
                    health_check_interval=comp_data["health_check_interval"],
                    metadata=comp_data["metadata"]
                )
                self._components[name] = component
            
            logger.info("System state loaded from disk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def reset_failsafe(self, force: bool = False) -> bool:
        """
        Reset failsafe state.
        
        Args:
            force: Force reset even if system is not in safe state
            
        Returns:
            True if failsafe reset successfully
        """
        with self._state_lock:
            if not force and self._state == SystemState.EMERGENCY_STOPPED:
                logger.error("Cannot reset failsafe while in emergency stop")
                return False
            
            self._failsafe_active = False
            logger.info("Failsafe reset")
            return True
    
    def shutdown(self, timeout: float = 10.0):
        """
        Shutdown the controller completely.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down trading system controller")
        
        # Stop system if active
        if self._state not in [SystemState.INACTIVE, SystemState.EMERGENCY_STOPPED]:
            self.stop_system(timeout / 2)
        
        # Stop monitoring
        self._stop_monitoring()
        
        # Shutdown executor
        self._executor.shutdown(wait=True, timeout=timeout / 2)
        
        # Persist final state
        self._persist_state()
        
        logger.info("Trading system controller shutdown complete")