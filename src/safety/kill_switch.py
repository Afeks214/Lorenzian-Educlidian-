"""
Multi-layered Kill Switch Architecture for Trading System Emergency Shutdown

Implements comprehensive emergency shutdown protocols with human override capabilities.
Provides multiple failsafe layers to ensure system can be safely halted under any conditions.
"""

import threading
import time
import logging
import json
import signal
import sys
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
import os

logger = logging.getLogger(__name__)

class ShutdownReason(Enum):
    """Reasons for emergency shutdown."""
    MANUAL_OVERRIDE = "manual_override"
    SYSTEM_ERROR = "system_error"
    RISK_THRESHOLD = "risk_threshold"
    MARKET_ANOMALY = "market_anomaly"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SIGNAL = "external_signal"
    VALIDATION_FAILURE = "validation_failure"

class ShutdownLevel(Enum):
    """Levels of shutdown severity."""
    GRACEFUL = "graceful"          # Normal shutdown with cleanup
    IMMEDIATE = "immediate"        # Fast shutdown, minimal cleanup
    EMERGENCY = "emergency"        # Instant shutdown, no cleanup
    FORCE_KILL = "force_kill"      # System kill, last resort

@dataclass
class ShutdownEvent:
    """Emergency shutdown event record."""
    timestamp: datetime
    reason: ShutdownReason
    level: ShutdownLevel
    initiator: str
    message: str
    system_state: Dict[str, Any]
    human_override: bool = False

class KillSwitchCore:
    """Core kill switch mechanism with multiple failsafe layers."""
    
    def __init__(self):
        self.shutdown_active = False
        self.shutdown_event = None
        self.shutdown_callbacks = []
        self.human_override_active = False
        self.lock = threading.RLock()
        
        # Emergency shutdown triggers
        self.emergency_triggers = {
            'manual_file': '/tmp/trading_emergency_stop',
            'memory_threshold': 0.95,  # 95% memory usage
            'cpu_threshold': 0.98,     # 98% CPU usage
            'error_count_threshold': 10,  # 10 consecutive errors
        }
        
        # Tracking
        self.consecutive_errors = 0
        self.last_health_check = time.time()
        self.shutdown_history = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("Kill switch core initialized")
    
    def _setup_signal_handlers(self):
        """Setup system signal handlers for emergency shutdown."""
        def signal_handler(signum, frame):
            signal_names = {
                signal.SIGTERM: "SIGTERM",
                signal.SIGINT: "SIGINT",
                signal.SIGUSR1: "SIGUSR1",
                signal.SIGUSR2: "SIGUSR2"
            }
            
            reason = ShutdownReason.EXTERNAL_SIGNAL
            level = ShutdownLevel.IMMEDIATE if signum == signal.SIGTERM else ShutdownLevel.GRACEFUL
            
            self.trigger_shutdown(
                reason=reason,
                level=level,
                initiator=f"signal_{signal_names.get(signum, signum)}",
                message=f"Received {signal_names.get(signum, signum)} signal"
            )
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGUSR2, signal_handler)
    
    def trigger_shutdown(
        self,
        reason: ShutdownReason,
        level: ShutdownLevel = ShutdownLevel.GRACEFUL,
        initiator: str = "unknown",
        message: str = "",
        human_override: bool = False
    ) -> bool:
        """Trigger emergency shutdown."""
        with self.lock:
            if self.shutdown_active:
                logger.warning(f"Shutdown already active, ignoring trigger from {initiator}")
                return False
            
            self.shutdown_active = True
            self.human_override_active = human_override
            
            # Create shutdown event
            self.shutdown_event = ShutdownEvent(
                timestamp=datetime.now(),
                reason=reason,
                level=level,
                initiator=initiator,
                message=message,
                system_state=self._capture_system_state(),
                human_override=human_override
            )
            
            # Log the emergency shutdown
            logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason.value} by {initiator} - {message}")
            
            # Execute shutdown
            self._execute_shutdown()
            
            return True
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for debugging."""
        try:
            process = psutil.Process(os.getpid())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'process_id': os.getpid(),
                'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads(),
                'system_memory': psutil.virtual_memory().percent,
                'system_cpu': psutil.cpu_percent(),
                'consecutive_errors': self.consecutive_errors,
                'last_health_check': self.last_health_check
            }
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {'error': str(e)}
    
    def _execute_shutdown(self):
        """Execute the emergency shutdown based on level."""
        if not self.shutdown_event:
            return
        
        level = self.shutdown_event.level
        
        try:
            if level == ShutdownLevel.GRACEFUL:
                self._graceful_shutdown()
            elif level == ShutdownLevel.IMMEDIATE:
                self._immediate_shutdown()
            elif level == ShutdownLevel.EMERGENCY:
                self._emergency_shutdown()
            elif level == ShutdownLevel.FORCE_KILL:
                self._force_kill_shutdown()
                
        except Exception as e:
            logger.error(f"Shutdown execution failed: {e}")
            # Escalate to force kill
            self._force_kill_shutdown()
    
    def _graceful_shutdown(self):
        """Graceful shutdown with full cleanup."""
        logger.info("Executing graceful shutdown...")
        
        # Save shutdown event
        self._save_shutdown_event()
        
        # Execute callbacks in reverse order
        for callback in reversed(self.shutdown_callbacks):
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        
        # Allow time for cleanup
        time.sleep(2)
        
        logger.info("Graceful shutdown complete")
        sys.exit(0)
    
    def _immediate_shutdown(self):
        """Immediate shutdown with minimal cleanup."""
        logger.info("Executing immediate shutdown...")
        
        # Save shutdown event
        self._save_shutdown_event()
        
        # Execute only critical callbacks
        for callback in reversed(self.shutdown_callbacks[:3]):  # Only first 3
            try:
                callback()
            except Exception as e:
                logger.error(f"Critical shutdown callback failed: {e}")
        
        # Short cleanup time
        time.sleep(0.5)
        
        logger.info("Immediate shutdown complete")
        sys.exit(1)
    
    def _emergency_shutdown(self):
        """Emergency shutdown with no cleanup."""
        logger.critical("Executing emergency shutdown...")
        
        # Try to save event quickly
        try:
            self._save_shutdown_event()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        logger.critical("Emergency shutdown complete")
        sys.exit(2)
    
    def _force_kill_shutdown(self):
        """Force kill shutdown - last resort."""
        logger.critical("Executing force kill shutdown...")
        
        # Kill process immediately
        os.kill(os.getpid(), signal.SIGKILL)
    
    def _save_shutdown_event(self):
        """Save shutdown event to file."""
        try:
            shutdown_file = '/tmp/trading_shutdown_event.json'
            with open(shutdown_file, 'w') as f:
                json.dump(asdict(self.shutdown_event), f, indent=2, default=str)
            
            # Also add to history
            self.shutdown_history.append(self.shutdown_event)
            
        except Exception as e:
            logger.error(f"Failed to save shutdown event: {e}")
    
    def register_shutdown_callback(self, callback: Callable):
        """Register callback to be executed on shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def check_emergency_triggers(self) -> bool:
        """Check for emergency triggers."""
        with self.lock:
            if self.shutdown_active:
                return True
            
            # Check manual file trigger
            if os.path.exists(self.emergency_triggers['manual_file']):
                self.trigger_shutdown(
                    reason=ShutdownReason.MANUAL_OVERRIDE,
                    level=ShutdownLevel.IMMEDIATE,
                    initiator="manual_file",
                    message="Manual emergency file detected",
                    human_override=True
                )
                return True
            
            # Check resource thresholds
            try:
                memory_percent = psutil.virtual_memory().percent / 100
                cpu_percent = psutil.cpu_percent() / 100
                
                if memory_percent > self.emergency_triggers['memory_threshold']:
                    self.trigger_shutdown(
                        reason=ShutdownReason.RESOURCE_EXHAUSTION,
                        level=ShutdownLevel.EMERGENCY,
                        initiator="memory_monitor",
                        message=f"Memory usage {memory_percent:.1%} exceeds threshold"
                    )
                    return True
                
                if cpu_percent > self.emergency_triggers['cpu_threshold']:
                    self.trigger_shutdown(
                        reason=ShutdownReason.RESOURCE_EXHAUSTION,
                        level=ShutdownLevel.EMERGENCY,
                        initiator="cpu_monitor",
                        message=f"CPU usage {cpu_percent:.1%} exceeds threshold"
                    )
                    return True
                
            except Exception as e:
                logger.error(f"Resource monitoring failed: {e}")
            
            # Check error count
            if self.consecutive_errors >= self.emergency_triggers['error_count_threshold']:
                self.trigger_shutdown(
                    reason=ShutdownReason.SYSTEM_ERROR,
                    level=ShutdownLevel.IMMEDIATE,
                    initiator="error_monitor",
                    message=f"Consecutive errors {self.consecutive_errors} exceeds threshold"
                )
                return True
            
            return False
    
    def report_error(self, error: Exception):
        """Report system error for tracking."""
        with self.lock:
            self.consecutive_errors += 1
            logger.error(f"System error reported: {error} (consecutive: {self.consecutive_errors})")
    
    def report_success(self):
        """Report successful operation to reset error count."""
        with self.lock:
            self.consecutive_errors = 0
    
    def is_shutdown_active(self) -> bool:
        """Check if shutdown is active."""
        return self.shutdown_active
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        with self.lock:
            return {
                'shutdown_active': self.shutdown_active,
                'human_override_active': self.human_override_active,
                'consecutive_errors': self.consecutive_errors,
                'last_health_check': self.last_health_check,
                'shutdown_event': asdict(self.shutdown_event) if self.shutdown_event else None
            }

class TradingSystemKillSwitch:
    """High-level kill switch for trading system with human override."""
    
    def __init__(self, trading_system=None):
        self.trading_system = trading_system
        self.kill_switch_core = KillSwitchCore()
        self.monitor_thread = None
        self.monitoring_active = False
        
        # Register system-specific shutdown callbacks
        self._register_trading_callbacks()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Trading system kill switch initialized")
    
    def _register_trading_callbacks(self):
        """Register trading-specific shutdown callbacks."""
        if self.trading_system:
            # Close positions callback
            def close_positions():
                try:
                    if hasattr(self.trading_system, 'close_all_positions'):
                        self.trading_system.close_all_positions()
                    logger.info("All positions closed")
                except Exception as e:
                    logger.error(f"Failed to close positions: {e}")
            
            # Stop agents callback
            def stop_agents():
                try:
                    if hasattr(self.trading_system, 'stop_agents'):
                        self.trading_system.stop_agents()
                    logger.info("All agents stopped")
                except Exception as e:
                    logger.error(f"Failed to stop agents: {e}")
            
            # Save state callback
            def save_state():
                try:
                    if hasattr(self.trading_system, 'save_state'):
                        self.trading_system.save_state()
                    logger.info("System state saved")
                except Exception as e:
                    logger.error(f"Failed to save state: {e}")
            
            # Register callbacks in order of priority
            self.kill_switch_core.register_shutdown_callback(close_positions)
            self.kill_switch_core.register_shutdown_callback(stop_agents)
            self.kill_switch_core.register_shutdown_callback(save_state)
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Kill switch monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        logger.info("Kill switch monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Check emergency triggers
                if self.kill_switch_core.check_emergency_triggers():
                    break
                
                # Update health check timestamp
                self.kill_switch_core.last_health_check = time.time()
                
                # Sleep before next check
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.kill_switch_core.report_error(e)
                time.sleep(5)  # Longer sleep on error
    
    def emergency_stop(self, reason: str = "", human_override: bool = False):
        """Trigger emergency stop with human override."""
        return self.kill_switch_core.trigger_shutdown(
            reason=ShutdownReason.MANUAL_OVERRIDE,
            level=ShutdownLevel.IMMEDIATE,
            initiator="human_operator",
            message=f"Human emergency stop: {reason}",
            human_override=human_override
        )
    
    def graceful_stop(self, reason: str = ""):
        """Trigger graceful shutdown."""
        return self.kill_switch_core.trigger_shutdown(
            reason=ShutdownReason.MANUAL_OVERRIDE,
            level=ShutdownLevel.GRACEFUL,
            initiator="operator",
            message=f"Graceful stop: {reason}"
        )
    
    def force_kill(self, reason: str = ""):
        """Force kill the system - last resort."""
        return self.kill_switch_core.trigger_shutdown(
            reason=ShutdownReason.MANUAL_OVERRIDE,
            level=ShutdownLevel.FORCE_KILL,
            initiator="operator",
            message=f"Force kill: {reason}"
        )
    
    def report_error(self, error: Exception):
        """Report system error."""
        self.kill_switch_core.report_error(error)
    
    def report_success(self):
        """Report successful operation."""
        self.kill_switch_core.report_success()
    
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self.kill_switch_core.is_shutdown_active()
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status."""
        return self.kill_switch_core.get_shutdown_status()

# Global kill switch instance
_global_kill_switch = None

def initialize_kill_switch(trading_system=None) -> TradingSystemKillSwitch:
    """Initialize global kill switch."""
    global _global_kill_switch
    _global_kill_switch = TradingSystemKillSwitch(trading_system)
    return _global_kill_switch

def get_kill_switch() -> Optional[TradingSystemKillSwitch]:
    """Get global kill switch instance."""
    return _global_kill_switch

def emergency_stop(reason: str = "", human_override: bool = False) -> bool:
    """Global emergency stop function."""
    if _global_kill_switch:
        return _global_kill_switch.emergency_stop(reason, human_override)
    else:
        logger.error("Kill switch not initialized")
        return False

def create_manual_stop_file():
    """Create manual stop file for emergency shutdown."""
    try:
        stop_file = '/tmp/trading_emergency_stop'
        with open(stop_file, 'w') as f:
            f.write(f"Manual emergency stop triggered at {datetime.now()}\n")
        logger.info(f"Manual stop file created: {stop_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create manual stop file: {e}")
        return False

if __name__ == "__main__":
    # Test the kill switch
    print("Testing kill switch...")
    
    kill_switch = initialize_kill_switch()
    
    # Test graceful shutdown
    print("Testing graceful shutdown in 3 seconds...")
    time.sleep(3)
    
    kill_switch.graceful_stop("Test shutdown")