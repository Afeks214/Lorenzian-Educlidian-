"""
TradingSystemController - Master Switch Controller for GrandModel Trading System

This module implements the foundational master switch controller that provides:
- Thread-safe state management with persistent storage
- JSON-based status file with corruption recovery
- Graceful shutdown with position protection
- Event-driven state changes with notifications
- Health monitoring and automatic failsafe
- Performance optimization with <1ms overhead per check
- Emergency stop functionality

Author: GrandModel Agent 1
"""

import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
from pathlib import Path
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System state enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemStatus:
    """System status data structure"""
    state: SystemState
    health: HealthStatus
    timestamp: float
    uptime: float
    last_heartbeat: float
    active_positions: int
    total_trades: int
    error_count: int
    last_error: Optional[str] = None
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class StateChangeEvent:
    """State change event data structure"""
    timestamp: float
    old_state: SystemState
    new_state: SystemState
    reason: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TradingSystemController:
    """
    Thread-safe master switch controller for the GrandModel trading system.
    
    Provides centralized state management, health monitoring, and emergency controls
    with high performance and reliability guarantees.
    """
    
    def __init__(
        self,
        state_file: str = "system_state.json",
        backup_file: str = "system_state.backup.json",
        health_check_interval: float = 5.0,
        max_error_count: int = 10,
        performance_cache_size: int = 1000
    ):
        """
        Initialize the TradingSystemController.
        
        Args:
            state_file: Path to JSON state file
            backup_file: Path to backup state file
            health_check_interval: Health check interval in seconds
            max_error_count: Maximum errors before failsafe
            performance_cache_size: Size of performance cache
        """
        self.state_file = Path(state_file)
        self.backup_file = Path(backup_file)
        self.health_check_interval = health_check_interval
        self.max_error_count = max_error_count
        
        # Thread safety
        self._lock = threading.RLock()
        self._state_lock = threading.RLock()
        
        # State management
        self._current_state = SystemState.INITIALIZING
        self._health_status = HealthStatus.UNKNOWN
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._error_count = 0
        self._last_error = None
        self._active_positions = 0
        self._total_trades = 0
        self._custom_metrics = {}
        
        # Event system
        self._state_change_callbacks: List[Callable[[StateChangeEvent], None]] = []
        self._health_callbacks: List[Callable[[HealthStatus], None]] = []
        self._emergency_callbacks: List[Callable[[], None]] = []
        
        # Performance optimization
        self._performance_cache_size = performance_cache_size
        self._cached_status = None
        self._cache_timestamp = 0
        self._cache_ttl = 0.001  # 1ms cache TTL for performance
        
        # Health monitoring
        self._health_monitor_thread = None
        self._health_monitor_active = False
        self._health_history = []
        self._max_health_history = 100
        
        # Shutdown management
        self._shutdown_requested = False
        self._shutdown_callbacks: List[Callable[[], None]] = []
        self._emergency_stop_active = False
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="SystemController")
        
        # Initialize system
        self._initialize_system()
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _initialize_system(self):
        """Initialize the system controller"""
        try:
            # Load previous state if available
            self._load_state()
            
            # Start health monitoring
            self._start_health_monitoring()
            
            # Set initial state
            self._transition_state(SystemState.RUNNING, "System initialized")
            
            logger.info("TradingSystemController initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system controller: {e}")
            self._transition_state(SystemState.ERROR, f"Initialization failed: {e}")
            raise
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown(graceful=True)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @contextmanager
    def _performance_timer(self, operation: str):
        """Context manager for performance monitoring"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            if elapsed > 1.0:  # Log if operation takes more than 1ms
                logger.warning(f"Performance warning: {operation} took {elapsed:.2f}ms")
    
    def get_state(self) -> SystemState:
        """
        Get current system state with performance optimization.
        
        Returns:
            Current system state
        """
        with self._performance_timer("get_state"):
            # Use cached state if available and fresh
            current_time = time.time()
            if (self._cached_status and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._cached_status.state
            
            with self._state_lock:
                return self._current_state
    
    def get_status(self) -> SystemStatus:
        """
        Get comprehensive system status with caching.
        
        Returns:
            SystemStatus object with current system information
        """
        with self._performance_timer("get_status"):
            current_time = time.time()
            
            # Check cache first
            if (self._cached_status and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._cached_status
            
            with self._state_lock:
                status = SystemStatus(
                    state=self._current_state,
                    health=self._health_status,
                    timestamp=current_time,
                    uptime=current_time - self._start_time,
                    last_heartbeat=self._last_heartbeat,
                    active_positions=self._active_positions,
                    total_trades=self._total_trades,
                    error_count=self._error_count,
                    last_error=self._last_error,
                    custom_metrics=self._custom_metrics.copy()
                )
                
                # Update cache
                self._cached_status = status
                self._cache_timestamp = current_time
                
                return status
    
    def _transition_state(self, new_state: SystemState, reason: str, metadata: Dict[str, Any] = None):
        """
        Thread-safe state transition with event notifications.
        
        Args:
            new_state: Target state
            reason: Reason for state change
            metadata: Additional metadata
        """
        with self._state_lock:
            old_state = self._current_state
            
            if old_state == new_state:
                return  # No change needed
            
            self._current_state = new_state
            
            # Update health status based on state
            if new_state == SystemState.ERROR:
                self._health_status = HealthStatus.CRITICAL
                self._error_count += 1
            elif new_state == SystemState.EMERGENCY_STOP:
                self._health_status = HealthStatus.CRITICAL
            elif new_state == SystemState.RUNNING:
                self._health_status = HealthStatus.HEALTHY
            
            # Create state change event
            event = StateChangeEvent(
                timestamp=time.time(),
                old_state=old_state,
                new_state=new_state,
                reason=reason,
                metadata=metadata or {}
            )
            
            # Invalidate cache
            self._cached_status = None
            
            # Persist state change
            self._save_state()
            
            # Notify callbacks asynchronously
            self._executor.submit(self._notify_state_change, event)
            
            logger.info(f"State transition: {old_state.value} -> {new_state.value}, reason: {reason}")
    
    def _notify_state_change(self, event: StateChangeEvent):
        """Notify all state change callbacks"""
        for callback in self._state_change_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def _notify_health_change(self, health_status: HealthStatus):
        """Notify all health change callbacks"""
        for callback in self._health_callbacks:
            try:
                callback(health_status)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")
    
    def _notify_emergency_stop(self):
        """Notify all emergency stop callbacks"""
        for callback in self._emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in emergency stop callback: {e}")
    
    def add_state_change_callback(self, callback: Callable[[StateChangeEvent], None]):
        """Add state change callback"""
        with self._lock:
            self._state_change_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[HealthStatus], None]):
        """Add health status callback"""
        with self._lock:
            self._health_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[], None]):
        """Add emergency stop callback"""
        with self._lock:
            self._emergency_callbacks.append(callback)
    
    def add_shutdown_callback(self, callback: Callable[[], None]):
        """Add shutdown callback"""
        with self._lock:
            self._shutdown_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove callback from all lists"""
        with self._lock:
            callback_lists = [
                self._state_change_callbacks,
                self._health_callbacks,
                self._emergency_callbacks,
                self._shutdown_callbacks
            ]
            
            for callback_list in callback_lists:
                if callback in callback_list:
                    callback_list.remove(callback)
    
    def _load_state(self):
        """Load state from JSON file with corruption recovery"""
        try:
            # Try to load from primary state file
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._restore_state_from_data(data)
                logger.info("State loaded from primary file")
                return
        except Exception as e:
            logger.warning(f"Failed to load primary state file: {e}")
        
        try:
            # Try to load from backup file
            if self.backup_file.exists():
                with open(self.backup_file, 'r') as f:
                    data = json.load(f)
                    self._restore_state_from_data(data)
                logger.info("State loaded from backup file")
                return
        except Exception as e:
            logger.warning(f"Failed to load backup state file: {e}")
        
        # If both files fail, start with default state
        logger.info("Starting with default state")
        self._reset_to_default_state()
    
    def _restore_state_from_data(self, data: Dict):
        """Restore state from loaded data"""
        try:
            # Restore basic state
            self._current_state = SystemState(data.get('state', SystemState.INITIALIZING.value))
            self._health_status = HealthStatus(data.get('health', HealthStatus.UNKNOWN.value))
            self._error_count = data.get('error_count', 0)
            self._last_error = data.get('last_error')
            self._active_positions = data.get('active_positions', 0)
            self._total_trades = data.get('total_trades', 0)
            self._custom_metrics = data.get('custom_metrics', {})
            
            # Restore timestamps
            self._start_time = data.get('start_time', time.time())
            self._last_heartbeat = data.get('last_heartbeat', time.time())
            
        except Exception as e:
            logger.error(f"Error restoring state: {e}")
            self._reset_to_default_state()
    
    def _reset_to_default_state(self):
        """Reset to default state"""
        self._current_state = SystemState.INITIALIZING
        self._health_status = HealthStatus.UNKNOWN
        self._error_count = 0
        self._last_error = None
        self._active_positions = 0
        self._total_trades = 0
        self._custom_metrics = {}
        self._start_time = time.time()
        self._last_heartbeat = time.time()
    
    def _save_state(self):
        """Save state to JSON file with atomic writes"""
        try:
            current_time = time.time()
            
            # Prepare state data
            state_data = {
                'state': self._current_state.value,
                'health': self._health_status.value,
                'timestamp': current_time,
                'start_time': self._start_time,
                'last_heartbeat': self._last_heartbeat,
                'error_count': self._error_count,
                'last_error': self._last_error,
                'active_positions': self._active_positions,
                'total_trades': self._total_trades,
                'custom_metrics': self._custom_metrics
            }
            
            # Atomic write to temporary file first
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Create backup of current file
            if self.state_file.exists():
                shutil.copy2(self.state_file, self.backup_file)
            
            # Atomically replace the state file
            temp_file.replace(self.state_file)
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self._health_monitor_thread is None or not self._health_monitor_thread.is_alive():
            self._health_monitor_active = True
            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True,
                name="HealthMonitor"
            )
            self._health_monitor_thread.start()
            logger.info("Health monitoring started")
    
    def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self._health_monitor_active and not self._shutdown_requested:
            try:
                # Update heartbeat
                self._last_heartbeat = time.time()
                
                # Check system health
                health_status = self._check_system_health()
                
                # Update health status if changed
                if health_status != self._health_status:
                    old_health = self._health_status
                    self._health_status = health_status
                    
                    # Notify callbacks
                    self._executor.submit(self._notify_health_change, health_status)
                    
                    logger.info(f"Health status changed: {old_health.value} -> {health_status.value}")
                
                # Store health history
                self._health_history.append({
                    'timestamp': time.time(),
                    'health': health_status.value,
                    'error_count': self._error_count
                })
                
                # Limit history size
                if len(self._health_history) > self._max_health_history:
                    self._health_history.pop(0)
                
                # Check for automatic failsafe
                if self._should_trigger_failsafe():
                    self._trigger_failsafe()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_system_health(self) -> HealthStatus:
        """Check system health and return status"""
        try:
            # Check error count
            if self._error_count >= self.max_error_count:
                return HealthStatus.CRITICAL
            
            # Check if in error state
            if self._current_state == SystemState.ERROR:
                return HealthStatus.CRITICAL
            
            # Check if emergency stop is active
            if self._emergency_stop_active:
                return HealthStatus.CRITICAL
            
            # Check heartbeat freshness
            time_since_heartbeat = time.time() - self._last_heartbeat
            if time_since_heartbeat > self.health_check_interval * 3:
                return HealthStatus.WARNING
            
            # Check if system is running normally
            if self._current_state == SystemState.RUNNING:
                return HealthStatus.HEALTHY
            
            return HealthStatus.WARNING
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return HealthStatus.UNKNOWN
    
    def _should_trigger_failsafe(self) -> bool:
        """Check if automatic failsafe should be triggered"""
        # Trigger failsafe if error count exceeds threshold
        if self._error_count >= self.max_error_count:
            return True
        
        # Trigger failsafe if health has been critical for too long
        if (self._health_status == HealthStatus.CRITICAL and 
            len(self._health_history) >= 3):
            recent_health = self._health_history[-3:]
            if all(h['health'] == HealthStatus.CRITICAL.value for h in recent_health):
                return True
        
        return False
    
    def _trigger_failsafe(self):
        """Trigger automatic failsafe"""
        logger.critical("Automatic failsafe triggered due to system health issues")
        self.emergency_stop("Automatic failsafe triggered")
    
    def pause(self, reason: str = "Manual pause"):
        """Pause the system"""
        with self._lock:
            if self._current_state == SystemState.RUNNING:
                self._transition_state(SystemState.PAUSED, reason)
                logger.info(f"System paused: {reason}")
            else:
                logger.warning(f"Cannot pause system from state: {self._current_state.value}")
    
    def resume(self, reason: str = "Manual resume"):
        """Resume the system"""
        with self._lock:
            if self._current_state == SystemState.PAUSED:
                self._transition_state(SystemState.RUNNING, reason)
                logger.info(f"System resumed: {reason}")
            else:
                logger.warning(f"Cannot resume system from state: {self._current_state.value}")
    
    def emergency_stop(self, reason: str = "Emergency stop"):
        """Emergency stop the system immediately"""
        with self._lock:
            logger.critical(f"EMERGENCY STOP: {reason}")
            
            self._emergency_stop_active = True
            self._transition_state(SystemState.EMERGENCY_STOP, reason)
            
            # Notify emergency callbacks immediately
            self._notify_emergency_stop()
            
            # Stop all trading activities
            self._stop_all_trading_activities()
    
    def _stop_all_trading_activities(self):
        """Stop all trading activities immediately"""
        try:
            # This would integrate with actual trading components
            # For now, just log the action
            logger.info("All trading activities stopped")
            
            # Reset position counts (in real implementation, this would close positions)
            self._active_positions = 0
            
        except Exception as e:
            logger.error(f"Error stopping trading activities: {e}")
    
    def shutdown(self, graceful: bool = True, timeout: float = 30.0):
        """
        Shutdown the system with optional graceful shutdown.
        
        Args:
            graceful: Whether to perform graceful shutdown
            timeout: Timeout for graceful shutdown in seconds
        """
        with self._lock:
            if self._shutdown_requested:
                return
            
            self._shutdown_requested = True
            
            logger.info(f"Initiating {'graceful' if graceful else 'immediate'} shutdown")
            
            if graceful:
                self._graceful_shutdown(timeout)
            else:
                self._immediate_shutdown()
    
    def _graceful_shutdown(self, timeout: float):
        """Perform graceful shutdown with position protection"""
        try:
            # Transition to stopping state
            self._transition_state(SystemState.STOPPING, "Graceful shutdown initiated")
            
            # Call shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in shutdown callback: {e}")
            
            # Wait for positions to be closed (simulation)
            start_time = time.time()
            while self._active_positions > 0 and time.time() - start_time < timeout:
                logger.info(f"Waiting for {self._active_positions} positions to close...")
                time.sleep(1)
            
            # Force close remaining positions if timeout reached
            if self._active_positions > 0:
                logger.warning(f"Forcing closure of {self._active_positions} remaining positions")
                self._active_positions = 0
            
            # Stop health monitoring
            self._health_monitor_active = False
            
            # Transition to stopped state
            self._transition_state(SystemState.STOPPED, "Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            self._immediate_shutdown()
    
    def _immediate_shutdown(self):
        """Perform immediate shutdown"""
        try:
            # Stop health monitoring
            self._health_monitor_active = False
            
            # Transition to stopped state
            self._transition_state(SystemState.STOPPED, "Immediate shutdown")
            
            # Force close all positions
            self._active_positions = 0
            
        except Exception as e:
            logger.error(f"Error during immediate shutdown: {e}")
    
    def reset_error_count(self):
        """Reset error count"""
        with self._lock:
            self._error_count = 0
            self._last_error = None
            logger.info("Error count reset")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update custom metrics"""
        with self._lock:
            self._custom_metrics.update(metrics)
            self._cached_status = None  # Invalidate cache
    
    def get_health_history(self) -> List[Dict]:
        """Get health history"""
        with self._lock:
            return self._health_history.copy()
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self._health_status == HealthStatus.HEALTHY
    
    def can_trade(self) -> bool:
        """Check if system can trade"""
        return (self._current_state == SystemState.RUNNING and 
                self._health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING] and
                not self._emergency_stop_active)
    
    def heartbeat(self):
        """Update heartbeat timestamp"""
        self._last_heartbeat = time.time()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown(graceful=True)
    
    def __del__(self):
        """Destructor"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except:
            pass


# Global instance (singleton pattern)
_system_controller = None
_controller_lock = threading.Lock()


def get_system_controller(**kwargs) -> TradingSystemController:
    """
    Get the global system controller instance (singleton).
    
    Args:
        **kwargs: Arguments for TradingSystemController initialization
    
    Returns:
        TradingSystemController instance
    """
    global _system_controller
    
    if _system_controller is None:
        with _controller_lock:
            if _system_controller is None:
                _system_controller = TradingSystemController(**kwargs)
    
    return _system_controller


def reset_system_controller():
    """Reset the global system controller (for testing)"""
    global _system_controller
    
    with _controller_lock:
        if _system_controller is not None:
            _system_controller.shutdown(graceful=False)
            _system_controller = None


# Convenience functions
def get_system_state() -> SystemState:
    """Get current system state"""
    return get_system_controller().get_state()


def get_system_status() -> SystemStatus:
    """Get current system status"""
    return get_system_controller().get_status()


def emergency_stop(reason: str = "Emergency stop"):
    """Emergency stop the system"""
    get_system_controller().emergency_stop(reason)


def is_system_healthy() -> bool:
    """Check if system is healthy"""
    return get_system_controller().is_healthy()


def can_system_trade() -> bool:
    """Check if system can trade"""
    return get_system_controller().can_trade()


if __name__ == "__main__":
    # Demo usage
    print("TradingSystemController Demo")
    print("=" * 50)
    
    # Create controller
    controller = TradingSystemController()
    
    # Add some callbacks
    def on_state_change(event: StateChangeEvent):
        print(f"State changed: {event.old_state.value} -> {event.new_state.value}")
    
    def on_health_change(health: HealthStatus):
        print(f"Health changed: {health.value}")
    
    controller.add_state_change_callback(on_state_change)
    controller.add_health_callback(on_health_change)
    
    # Test operations
    print(f"Initial state: {controller.get_state().value}")
    print(f"Can trade: {controller.can_trade()}")
    
    # Pause and resume
    controller.pause("Testing pause")
    time.sleep(1)
    controller.resume("Testing resume")
    
    # Update metrics
    controller.update_metrics({"test_metric": 42})
    
    # Get status
    status = controller.get_status()
    print(f"Status: {status.state.value}, Health: {status.health.value}")
    print(f"Uptime: {status.uptime:.2f}s")
    
    # Shutdown
    controller.shutdown(graceful=True)
    print("Controller shutdown complete")