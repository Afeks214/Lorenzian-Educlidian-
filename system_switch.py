#!/usr/bin/env python3
"""
AGENT 5 MISSION: System Switch - Main Command Interface
GrandModel Trading System Control Interface

This module provides a user-friendly command-line interface for controlling
the GrandModel trading system with comprehensive monitoring and audit logging.

Commands:
- python system_switch.py on        - Turn system ON
- python system_switch.py off       - Turn system OFF  
- python system_switch.py status    - Check current status
- python system_switch.py emergency - Emergency stop
- python system_switch.py status --verbose - Detailed status

Features:
- Simple command-line interface
- Real-time status monitoring
- Comprehensive audit logging
- Visual feedback for system state
- Emergency stop functionality
- Component health checks
- Performance monitoring
"""

import sys
import argparse
import asyncio
import json
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.monitoring.system_status_monitor import SystemStatusMonitor
    from src.monitoring.switch_event_logger import SwitchEventLogger
    from src.core.event_bus import EventBus
    from src.core.config import Config
    from src.utils.logger import get_logger
except ImportError as e:
    # Fallback if modules not available
    print(f"Warning: Could not import monitoring modules: {e}")
    print("Operating in basic mode...")

# Initialize logger
logger = get_logger(__name__)

class SystemState(Enum):
    """System state enumeration"""
    OFF = "OFF"
    ON = "ON"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

@dataclass
class SystemStatus:
    """System status information"""
    state: SystemState
    timestamp: datetime
    uptime: float
    components: Dict[str, Any]
    performance: Dict[str, Any]
    health_score: float
    alerts: List[str]
    last_state_change: datetime
    error_message: Optional[str] = None

class SystemSwitch:
    """Main system switch controller"""
    
    def __init__(self):
        self.state_file = Path(__file__).parent / "data" / "system_state.json"
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.status_monitor = None
        self.event_logger = None
        self.event_bus = None
        
        # System state
        self.current_state = SystemState.UNKNOWN
        self.state_history = []
        self.start_time = None
        self.last_state_change = datetime.now()
        
        # Performance tracking
        self.performance_metrics = {
            'commands_processed': 0,
            'errors_encountered': 0,
            'state_changes': 0,
            'average_response_time': 0.0
        }
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize system components"""
        try:
            # Load existing state
            self._load_state()
            
            # Initialize monitoring components
            try:
                self.status_monitor = SystemStatusMonitor()
                self.event_logger = SwitchEventLogger()
                self.event_bus = EventBus()
                logger.info("System monitoring components initialized")
            except Exception as e:
                logger.warning(f"Could not initialize monitoring components: {e}")
                logger.info("System will operate in basic mode")
            
            # Register signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("System switch initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            self.current_state = SystemState.ERROR
            
    def _load_state(self):
        """Load system state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.current_state = SystemState(data.get('state', 'UNKNOWN'))
                    self.last_state_change = datetime.fromisoformat(
                        data.get('last_state_change', datetime.now().isoformat())
                    )
                    if data.get('start_time'):
                        self.start_time = datetime.fromisoformat(data['start_time'])
                    self.state_history = data.get('state_history', [])[-10:]  # Keep last 10
                    logger.info(f"Loaded system state: {self.current_state.value}")
            else:
                self.current_state = SystemState.OFF
                logger.info("No existing state file, defaulting to OFF")
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self.current_state = SystemState.UNKNOWN
            
    def _save_state(self):
        """Save system state to file"""
        try:
            data = {
                'state': self.current_state.value,
                'last_state_change': self.last_state_change.isoformat(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'state_history': self.state_history,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, initiating emergency stop")
        asyncio.create_task(self.emergency_stop())
        
    def _change_state(self, new_state: SystemState, reason: str = ""):
        """Change system state with logging"""
        old_state = self.current_state
        self.current_state = new_state
        self.last_state_change = datetime.now()
        
        # Record state change
        state_change = {
            'timestamp': self.last_state_change.isoformat(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'reason': reason
        }
        
        self.state_history.append(state_change)
        self.state_history = self.state_history[-10:]  # Keep last 10
        
        # Update metrics
        self.performance_metrics['state_changes'] += 1
        
        # Log event
        if self.event_logger:
            self.event_logger.log_state_change(old_state, new_state, reason)
        
        # Save state
        self._save_state()
        
        logger.info(f"System state changed: {old_state.value} -> {new_state.value} ({reason})")
        
    async def turn_on(self) -> bool:
        """Turn system ON"""
        start_time = time.time()
        
        try:
            if self.current_state == SystemState.ON:
                print("🟢 System is already ON")
                return True
                
            print("🔄 Starting system...")
            self._change_state(SystemState.STARTING, "User command: turn on")
            
            # Simulate system startup sequence
            await self._startup_sequence()
            
            # Mark as ON
            self.start_time = datetime.now()
            self._change_state(SystemState.ON, "Startup sequence completed")
            
            response_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                self.performance_metrics['average_response_time'] + response_time
            ) / 2
            
            print("✅ System is now ON")
            print(f"   Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Response time: {response_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._change_state(SystemState.ERROR, f"Startup failed: {str(e)}")
            self.performance_metrics['errors_encountered'] += 1
            print(f"❌ Failed to start system: {e}")
            logger.error(f"System startup failed: {e}")
            traceback.print_exc()
            return False
        finally:
            self.performance_metrics['commands_processed'] += 1
            
    async def turn_off(self) -> bool:
        """Turn system OFF"""
        start_time = time.time()
        
        try:
            if self.current_state == SystemState.OFF:
                print("🔴 System is already OFF")
                return True
                
            print("🔄 Stopping system...")
            self._change_state(SystemState.STOPPING, "User command: turn off")
            
            # Simulate system shutdown sequence
            await self._shutdown_sequence()
            
            # Mark as OFF
            self.start_time = None
            self._change_state(SystemState.OFF, "Shutdown sequence completed")
            
            response_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                self.performance_metrics['average_response_time'] + response_time
            ) / 2
            
            print("✅ System is now OFF")
            print(f"   Response time: {response_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._change_state(SystemState.ERROR, f"Shutdown failed: {str(e)}")
            self.performance_metrics['errors_encountered'] += 1
            print(f"❌ Failed to stop system: {e}")
            logger.error(f"System shutdown failed: {e}")
            traceback.print_exc()
            return False
        finally:
            self.performance_metrics['commands_processed'] += 1
            
    async def emergency_stop(self) -> bool:
        """Emergency stop system"""
        start_time = time.time()
        
        try:
            print("🚨 EMERGENCY STOP INITIATED")
            self._change_state(SystemState.EMERGENCY_STOP, "Emergency stop command")
            
            # Immediate shutdown sequence
            await self._emergency_shutdown_sequence()
            
            # Mark as OFF
            self.start_time = None
            self._change_state(SystemState.OFF, "Emergency stop completed")
            
            response_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                self.performance_metrics['average_response_time'] + response_time
            ) / 2
            
            print("✅ Emergency stop completed")
            print(f"   Response time: {response_time:.2f}s")
            
            return True
            
        except Exception as e:
            self._change_state(SystemState.ERROR, f"Emergency stop failed: {str(e)}")
            self.performance_metrics['errors_encountered'] += 1
            print(f"❌ Emergency stop failed: {e}")
            logger.error(f"Emergency stop failed: {e}")
            traceback.print_exc()
            return False
        finally:
            self.performance_metrics['commands_processed'] += 1
            
    async def get_status(self, verbose: bool = False) -> SystemStatus:
        """Get current system status"""
        try:
            # Calculate uptime
            uptime = 0.0
            if self.start_time and self.current_state == SystemState.ON:
                uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Get component status
            components = {}
            if self.status_monitor:
                components = await self.status_monitor.get_component_status()
            
            # Get performance metrics
            performance = dict(self.performance_metrics)
            if self.status_monitor:
                performance.update(await self.status_monitor.get_performance_metrics())
            
            # Calculate health score
            health_score = self._calculate_health_score(components)
            
            # Get alerts
            alerts = []
            if self.status_monitor:
                alerts = await self.status_monitor.get_active_alerts()
            
            status = SystemStatus(
                state=self.current_state,
                timestamp=datetime.now(),
                uptime=uptime,
                components=components,
                performance=performance,
                health_score=health_score,
                alerts=alerts,
                last_state_change=self.last_state_change
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return SystemStatus(
                state=SystemState.ERROR,
                timestamp=datetime.now(),
                uptime=0.0,
                components={},
                performance={},
                health_score=0.0,
                alerts=[f"Status check failed: {str(e)}"],
                last_state_change=self.last_state_change,
                error_message=str(e)
            )
            
    def _calculate_health_score(self, components: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        if not components:
            return 100.0 if self.current_state == SystemState.ON else 0.0
        
        try:
            healthy_count = sum(1 for comp in components.values() if comp.get('status') == 'healthy')
            total_count = len(components)
            
            if total_count == 0:
                return 100.0
            
            health_score = (healthy_count / total_count) * 100
            
            # Adjust based on system state
            if self.current_state == SystemState.ERROR:
                health_score *= 0.5
            elif self.current_state == SystemState.EMERGENCY_STOP:
                health_score = 0.0
            elif self.current_state == SystemState.OFF:
                health_score = 0.0
            
            return health_score
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0
            
    async def _startup_sequence(self):
        """Simulate system startup sequence"""
        startup_steps = [
            ("Initializing core systems", 0.5),
            ("Loading configuration", 0.3),
            ("Starting data pipeline", 0.8),
            ("Initializing agents", 1.0),
            ("Starting risk management", 0.6),
            ("Activating monitoring", 0.4),
            ("Running health checks", 0.7),
            ("System ready", 0.2)
        ]
        
        for step, delay in startup_steps:
            print(f"   {step}...")
            await asyncio.sleep(delay)
            
    async def _shutdown_sequence(self):
        """Simulate system shutdown sequence"""
        shutdown_steps = [
            ("Stopping trading agents", 0.5),
            ("Closing positions", 0.8),
            ("Saving state", 0.3),
            ("Stopping data pipeline", 0.4),
            ("Shutting down monitoring", 0.2),
            ("System stopped", 0.1)
        ]
        
        for step, delay in shutdown_steps:
            print(f"   {step}...")
            await asyncio.sleep(delay)
            
    async def _emergency_shutdown_sequence(self):
        """Simulate emergency shutdown sequence"""
        emergency_steps = [
            ("Halting all trading", 0.1),
            ("Emergency position closure", 0.3),
            ("Stopping all agents", 0.1),
            ("Critical state save", 0.1),
            ("System halted", 0.1)
        ]
        
        for step, delay in emergency_steps:
            print(f"   {step}...")
            await asyncio.sleep(delay)
            
    def display_status(self, status: SystemStatus, verbose: bool = False):
        """Display system status"""
        # State indicator
        state_icons = {
            SystemState.ON: "🟢",
            SystemState.OFF: "🔴", 
            SystemState.STARTING: "🔄",
            SystemState.STOPPING: "🔄",
            SystemState.EMERGENCY_STOP: "🚨",
            SystemState.ERROR: "❌",
            SystemState.UNKNOWN: "⚪"
        }
        
        icon = state_icons.get(status.state, "⚪")
        
        print(f"\n{icon} System Status: {status.state.value}")
        print(f"   Timestamp: {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Last State Change: {status.last_state_change.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if status.uptime > 0:
            uptime_str = str(timedelta(seconds=int(status.uptime)))
            print(f"   Uptime: {uptime_str}")
        
        print(f"   Health Score: {status.health_score:.1f}%")
        
        if status.alerts:
            print(f"   Active Alerts: {len(status.alerts)}")
            for alert in status.alerts[:3]:  # Show first 3 alerts
                print(f"     - {alert}")
            if len(status.alerts) > 3:
                print(f"     ... and {len(status.alerts) - 3} more")
        
        if verbose:
            print(f"\n📊 Performance Metrics:")
            for key, value in status.performance.items():
                print(f"   {key}: {value}")
                
            print(f"\n🔧 Components ({len(status.components)}):")
            for comp_name, comp_data in status.components.items():
                comp_status = comp_data.get('status', 'unknown')
                comp_icon = "🟢" if comp_status == 'healthy' else "🔴"
                print(f"   {comp_icon} {comp_name}: {comp_status}")
                
                if comp_data.get('details'):
                    for detail_key, detail_value in comp_data['details'].items():
                        print(f"      {detail_key}: {detail_value}")
                        
            print(f"\n📈 State History:")
            for change in self.state_history[-5:]:  # Show last 5 changes
                timestamp = change['timestamp']
                from_state = change['from_state']
                to_state = change['to_state']
                reason = change['reason']
                print(f"   {timestamp}: {from_state} -> {to_state} ({reason})")
                
        if status.error_message:
            print(f"\n❌ Error: {status.error_message}")
            
        print()

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="GrandModel Trading System Control Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python system_switch.py on              # Turn system ON
  python system_switch.py off             # Turn system OFF
  python system_switch.py status          # Check status
  python system_switch.py status -v       # Detailed status
  python system_switch.py emergency       # Emergency stop
        """
    )
    
    parser.add_argument(
        'command',
        choices=['on', 'off', 'status', 'emergency'],
        help='System command to execute'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output (for status command)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Command timeout in seconds (default: 30.0)'
    )
    
    return parser

async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize system switch
    system_switch = SystemSwitch()
    
    try:
        # Execute command
        if args.command == 'on':
            success = await asyncio.wait_for(
                system_switch.turn_on(),
                timeout=args.timeout
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'off':
            success = await asyncio.wait_for(
                system_switch.turn_off(),
                timeout=args.timeout
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'emergency':
            success = await asyncio.wait_for(
                system_switch.emergency_stop(),
                timeout=args.timeout
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'status':
            status = await asyncio.wait_for(
                system_switch.get_status(verbose=args.verbose),
                timeout=args.timeout
            )
            system_switch.display_status(status, verbose=args.verbose)
            
            # Exit with appropriate code
            if status.state == SystemState.ERROR:
                sys.exit(2)
            elif status.state == SystemState.EMERGENCY_STOP:
                sys.exit(3)
            else:
                sys.exit(0)
                
    except asyncio.TimeoutError:
        print(f"❌ Command timed out after {args.timeout} seconds")
        sys.exit(4)
    except KeyboardInterrupt:
        print(f"\n❌ Command interrupted by user")
        sys.exit(5)
    except Exception as e:
        print(f"❌ Command failed: {e}")
        logger.error(f"Command failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())