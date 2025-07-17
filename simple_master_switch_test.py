#!/usr/bin/env python3
"""
Simple test to verify the master switch integration works.

This script verifies that the decorator pattern is correctly implemented
and can block execution when the system is in emergency mode.
"""

import asyncio
import logging
import sys
import os
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockKillSwitch:
    """Mock kill switch for testing."""
    
    def __init__(self):
        self.active = False
    
    def is_active(self):
        return self.active
    
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False


class MockOperationalControls:
    """Mock operational controls for testing."""
    
    def __init__(self):
        self.emergency_stop = False
        self.maintenance_mode = False
    
    def trigger_emergency_stop(self):
        self.emergency_stop = True
    
    def reset_emergency_stop(self):
        self.emergency_stop = False
    
    def enable_maintenance_mode(self):
        self.maintenance_mode = True
    
    def disable_maintenance_mode(self):
        self.maintenance_mode = False


# Global mock instances
_mock_kill_switch = MockKillSwitch()
_mock_operational_controls = MockOperationalControls()


def get_kill_switch():
    """Mock get_kill_switch function."""
    return _mock_kill_switch


def require_system_active(func):
    """
    Decorator to ensure system is active before executing trading functions.
    
    Checks both kill switch and operational controls to ensure system safety.
    Blocks execution if system is in emergency stop or maintenance mode.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check kill switch first
        kill_switch = get_kill_switch()
        if kill_switch and kill_switch.is_active():
            logger.error(f"BLOCKED: {func.__name__} - Kill switch is active")
            # Don't execute the function, just return
            return
        
        # Check operational controls
        if hasattr(self, 'operational_controls') and self.operational_controls:
            if self.operational_controls.emergency_stop:
                logger.error(f"BLOCKED: {func.__name__} - Emergency stop is active")
                return
            
            if self.operational_controls.maintenance_mode:
                logger.warning(f"BLOCKED: {func.__name__} - System is in maintenance mode")
                return
        
        # System is active, proceed with execution
        return await func(self, *args, **kwargs)
    
    return wrapper


class TestExecutionEngine:
    """Test execution engine with safety controls."""
    
    def __init__(self, operational_controls=None):
        self.operational_controls = operational_controls
        self.executions_processed = 0
        self.running = False
    
    @require_system_active
    async def start_execution_listener(self):
        """Start execution listener."""
        self.running = True
        logger.info("🚀 Execution listener started")
    
    @require_system_active
    async def process_execution_event(self, event_data):
        """Process execution event."""
        self.executions_processed += 1
        logger.info(f"📥 Processing execution event: {event_data}")
    
    def get_stats(self):
        """Get execution statistics."""
        system_state = "active"
        kill_switch = get_kill_switch()
        if kill_switch and kill_switch.is_active():
            system_state = "kill_switch_active"
        elif self.operational_controls and self.operational_controls.emergency_stop:
            system_state = "emergency_stop"
        elif self.operational_controls and self.operational_controls.maintenance_mode:
            system_state = "maintenance_mode"
        
        return {
            'executions_processed': self.executions_processed,
            'running': self.running,
            'system_state': system_state
        }


class TestLiveExecutionHandler:
    """Test live execution handler with safety controls."""
    
    def __init__(self, config, operational_controls=None):
        self.config = config
        self.operational_controls = operational_controls
        self.total_orders = 0
        self.running = False
    
    @require_system_active
    async def start(self):
        """Start the handler."""
        self.running = True
        logger.info("🚀 Live execution handler started")
    
    @require_system_active
    async def execute_trade(self, trade_signal):
        """Execute a trade."""
        self.total_orders += 1
        logger.info(f"📤 Executing trade: {trade_signal}")
    
    def get_status(self):
        """Get handler status."""
        system_state = "active"
        kill_switch = get_kill_switch()
        if kill_switch and kill_switch.is_active():
            system_state = "kill_switch_active"
        elif self.operational_controls and self.operational_controls.emergency_stop:
            system_state = "emergency_stop"
        elif self.operational_controls and self.operational_controls.maintenance_mode:
            system_state = "maintenance_mode"
        
        return {
            'total_orders': self.total_orders,
            'running': self.running,
            'system_state': system_state
        }


async def test_normal_operation():
    """Test normal operation when system is active."""
    logger.info("Testing normal operation...")
    
    # Reset all controls to normal state
    _mock_kill_switch.deactivate()
    _mock_operational_controls.reset_emergency_stop()
    _mock_operational_controls.disable_maintenance_mode()
    
    # Create test components
    engine = TestExecutionEngine(_mock_operational_controls)
    handler = TestLiveExecutionHandler({}, _mock_operational_controls)
    
    # Test normal operations
    await engine.start_execution_listener()
    await engine.process_execution_event({"action": "buy", "symbol": "NQ"})
    
    await handler.start()
    await handler.execute_trade({"action": "BUY", "quantity": 1})
    
    # Check results
    engine_stats = engine.get_stats()
    handler_status = handler.get_status()
    
    logger.info(f"Engine stats: {engine_stats}")
    logger.info(f"Handler status: {handler_status}")
    
    assert engine_stats['system_state'] == 'active'
    assert handler_status['system_state'] == 'active'
    assert engine_stats['executions_processed'] == 1
    assert handler_status['total_orders'] == 1
    
    logger.info("✅ Normal operation test passed")
    return True


async def test_kill_switch_blocked():
    """Test blocked operation when kill switch is active."""
    logger.info("Testing kill switch blocked operation...")
    
    # Activate kill switch
    _mock_kill_switch.activate()
    
    # Create test components
    engine = TestExecutionEngine(_mock_operational_controls)
    handler = TestLiveExecutionHandler({}, _mock_operational_controls)
    
    # Test blocked operations
    await engine.start_execution_listener()
    await engine.process_execution_event({"action": "buy", "symbol": "NQ"})
    
    await handler.start()
    await handler.execute_trade({"action": "BUY", "quantity": 1})
    
    # Check results
    engine_stats = engine.get_stats()
    handler_status = handler.get_status()
    
    logger.info(f"Engine stats: {engine_stats}")
    logger.info(f"Handler status: {handler_status}")
    
    assert engine_stats['system_state'] == 'kill_switch_active'
    assert handler_status['system_state'] == 'kill_switch_active'
    assert engine_stats['executions_processed'] == 0  # Should be blocked
    assert handler_status['total_orders'] == 0  # Should be blocked
    
    logger.info("✅ Kill switch blocked operation test passed")
    return True


async def test_emergency_stop_blocked():
    """Test blocked operation when emergency stop is active."""
    logger.info("Testing emergency stop blocked operation...")
    
    # Reset kill switch but activate emergency stop
    _mock_kill_switch.deactivate()
    _mock_operational_controls.trigger_emergency_stop()
    
    # Create test components
    engine = TestExecutionEngine(_mock_operational_controls)
    handler = TestLiveExecutionHandler({}, _mock_operational_controls)
    
    # Test blocked operations
    await engine.start_execution_listener()
    await engine.process_execution_event({"action": "buy", "symbol": "NQ"})
    
    await handler.start()
    await handler.execute_trade({"action": "BUY", "quantity": 1})
    
    # Check results
    engine_stats = engine.get_stats()
    handler_status = handler.get_status()
    
    logger.info(f"Engine stats: {engine_stats}")
    logger.info(f"Handler status: {handler_status}")
    
    assert engine_stats['system_state'] == 'emergency_stop'
    assert handler_status['system_state'] == 'emergency_stop'
    assert engine_stats['executions_processed'] == 0  # Should be blocked
    assert handler_status['total_orders'] == 0  # Should be blocked
    
    logger.info("✅ Emergency stop blocked operation test passed")
    return True


async def test_maintenance_mode_blocked():
    """Test blocked operation when maintenance mode is active."""
    logger.info("Testing maintenance mode blocked operation...")
    
    # Reset emergency stop but activate maintenance mode
    _mock_operational_controls.reset_emergency_stop()
    _mock_operational_controls.enable_maintenance_mode()
    
    # Create test components
    engine = TestExecutionEngine(_mock_operational_controls)
    handler = TestLiveExecutionHandler({}, _mock_operational_controls)
    
    # Test blocked operations
    await engine.start_execution_listener()
    await engine.process_execution_event({"action": "buy", "symbol": "NQ"})
    
    await handler.start()
    await handler.execute_trade({"action": "BUY", "quantity": 1})
    
    # Check results
    engine_stats = engine.get_stats()
    handler_status = handler.get_status()
    
    logger.info(f"Engine stats: {engine_stats}")
    logger.info(f"Handler status: {handler_status}")
    
    assert engine_stats['system_state'] == 'maintenance_mode'
    assert handler_status['system_state'] == 'maintenance_mode'
    assert engine_stats['executions_processed'] == 0  # Should be blocked
    assert handler_status['total_orders'] == 0  # Should be blocked
    
    logger.info("✅ Maintenance mode blocked operation test passed")
    return True


async def run_all_tests():
    """Run all integration tests."""
    logger.info("Starting master switch integration tests...")
    
    tests = [
        test_normal_operation,
        test_kill_switch_blocked,
        test_emergency_stop_blocked,
        test_maintenance_mode_blocked
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All master switch integration tests passed!")
        return True
    else:
        logger.error(f"❌ {failed} tests failed")
        return False


async def main():
    """Main test runner."""
    logger.info("Starting master switch integration test suite...")
    
    success = await run_all_tests()
    
    if success:
        logger.info("✅ Master switch integration is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Master switch integration has issues!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())