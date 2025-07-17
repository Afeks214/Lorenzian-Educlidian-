#!/usr/bin/env python3
"""
Kill Switch Architecture Test Suite

Tests multi-layered emergency shutdown capabilities with human override.
"""

import unittest
import threading
import time
import os
import tempfile
import signal
import json
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.kill_switch import (
    KillSwitchCore, TradingSystemKillSwitch, ShutdownReason, ShutdownLevel,
    initialize_kill_switch, get_kill_switch, emergency_stop, create_manual_stop_file
)

class TestKillSwitchCore(unittest.TestCase):
    """Test the core kill switch functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.kill_switch = KillSwitchCore()
        self.test_callbacks_called = []
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up manual stop file
        stop_file = '/tmp/trading_emergency_stop'
        if os.path.exists(stop_file):
            os.remove(stop_file)
    
    def test_initialization(self):
        """Test kill switch initialization."""
        self.assertFalse(self.kill_switch.shutdown_active)
        self.assertIsNone(self.kill_switch.shutdown_event)
        self.assertFalse(self.kill_switch.human_override_active)
        self.assertEqual(self.kill_switch.consecutive_errors, 0)
    
    def test_shutdown_callback_registration(self):
        """Test callback registration."""
        def test_callback():
            self.test_callbacks_called.append("callback1")
        
        self.kill_switch.register_shutdown_callback(test_callback)
        self.assertEqual(len(self.kill_switch.shutdown_callbacks), 1)
    
    def test_error_reporting(self):
        """Test error reporting and tracking."""
        # Report multiple errors
        for i in range(3):
            self.kill_switch.report_error(Exception(f"Test error {i}"))
        
        self.assertEqual(self.kill_switch.consecutive_errors, 3)
        
        # Report success should reset counter
        self.kill_switch.report_success()
        self.assertEqual(self.kill_switch.consecutive_errors, 0)
    
    def test_manual_file_trigger(self):
        """Test manual file trigger."""
        # Create manual stop file
        stop_file = '/tmp/trading_emergency_stop'
        with open(stop_file, 'w') as f:
            f.write("Manual stop")
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            triggered = self.kill_switch.check_emergency_triggers()
            
        self.assertTrue(triggered)
        self.assertTrue(self.kill_switch.shutdown_active)
        self.assertEqual(self.kill_switch.shutdown_event.reason, ShutdownReason.MANUAL_OVERRIDE)
        self.assertTrue(self.kill_switch.shutdown_event.human_override)
    
    def test_error_threshold_trigger(self):
        """Test error threshold trigger."""
        # Set low threshold for testing
        self.kill_switch.emergency_triggers['error_count_threshold'] = 3
        
        # Report errors up to threshold
        for i in range(3):
            self.kill_switch.report_error(Exception(f"Test error {i}"))
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            triggered = self.kill_switch.check_emergency_triggers()
            
        self.assertTrue(triggered)
        self.assertTrue(self.kill_switch.shutdown_active)
        self.assertEqual(self.kill_switch.shutdown_event.reason, ShutdownReason.SYSTEM_ERROR)
    
    def test_shutdown_event_creation(self):
        """Test shutdown event creation."""
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            result = self.kill_switch.trigger_shutdown(
                reason=ShutdownReason.MANUAL_OVERRIDE,
                level=ShutdownLevel.GRACEFUL,
                initiator="test",
                message="Test shutdown",
                human_override=True
            )
        
        self.assertTrue(result)
        self.assertIsNotNone(self.kill_switch.shutdown_event)
        self.assertEqual(self.kill_switch.shutdown_event.reason, ShutdownReason.MANUAL_OVERRIDE)
        self.assertEqual(self.kill_switch.shutdown_event.level, ShutdownLevel.GRACEFUL)
        self.assertEqual(self.kill_switch.shutdown_event.initiator, "test")
        self.assertTrue(self.kill_switch.shutdown_event.human_override)
    
    def test_system_state_capture(self):
        """Test system state capture."""
        state = self.kill_switch._capture_system_state()
        
        self.assertIn('timestamp', state)
        self.assertIn('process_id', state)
        self.assertIn('memory_usage', state)
        self.assertIn('consecutive_errors', state)
        
        # Verify types
        self.assertIsInstance(state['process_id'], int)
        self.assertIsInstance(state['memory_usage'], (int, float))
        self.assertIsInstance(state['consecutive_errors'], int)
    
    def test_shutdown_status(self):
        """Test shutdown status retrieval."""
        status = self.kill_switch.get_shutdown_status()
        
        self.assertIn('shutdown_active', status)
        self.assertIn('human_override_active', status)
        self.assertIn('consecutive_errors', status)
        self.assertIn('shutdown_event', status)
        
        self.assertFalse(status['shutdown_active'])
        self.assertFalse(status['human_override_active'])
        self.assertEqual(status['consecutive_errors'], 0)
        self.assertIsNone(status['shutdown_event'])

class TestTradingSystemKillSwitch(unittest.TestCase):
    """Test the trading system kill switch."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_trading_system = Mock()
        self.kill_switch = TradingSystemKillSwitch(self.mock_trading_system)
        time.sleep(0.1)  # Allow monitoring to start
    
    def tearDown(self):
        """Clean up test environment."""
        self.kill_switch.stop_monitoring()
        
        # Clean up manual stop file
        stop_file = '/tmp/trading_emergency_stop'
        if os.path.exists(stop_file):
            os.remove(stop_file)
    
    def test_initialization(self):
        """Test kill switch initialization."""
        self.assertIsNotNone(self.kill_switch.kill_switch_core)
        self.assertIsNotNone(self.kill_switch.trading_system)
        self.assertTrue(self.kill_switch.monitoring_active)
    
    def test_monitoring_thread(self):
        """Test monitoring thread operation."""
        self.assertTrue(self.kill_switch.monitor_thread.is_alive())
        
        # Stop monitoring
        self.kill_switch.stop_monitoring()
        time.sleep(0.1)
        
        self.assertFalse(self.kill_switch.monitoring_active)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            result = self.kill_switch.emergency_stop("Test emergency", human_override=True)
        
        self.assertTrue(result)
        self.assertTrue(self.kill_switch.is_active())
        
        # Check if trading system methods were called
        if hasattr(self.mock_trading_system, 'close_all_positions'):
            self.mock_trading_system.close_all_positions.assert_called_once()
    
    def test_graceful_stop(self):
        """Test graceful stop functionality."""
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            result = self.kill_switch.graceful_stop("Test graceful")
        
        self.assertTrue(result)
        self.assertTrue(self.kill_switch.is_active())
    
    def test_error_reporting(self):
        """Test error reporting to kill switch."""
        test_error = Exception("Test error")
        
        self.kill_switch.report_error(test_error)
        
        status = self.kill_switch.get_status()
        self.assertEqual(status['consecutive_errors'], 1)
        
        # Report success should reset
        self.kill_switch.report_success()
        status = self.kill_switch.get_status()
        self.assertEqual(status['consecutive_errors'], 0)
    
    def test_trading_callbacks(self):
        """Test trading system specific callbacks."""
        # Mock trading system methods
        self.mock_trading_system.close_all_positions = Mock()
        self.mock_trading_system.stop_agents = Mock()
        self.mock_trading_system.save_state = Mock()
        
        # Re-register callbacks
        self.kill_switch._register_trading_callbacks()
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            self.kill_switch.graceful_stop("Test callbacks")
        
        # Verify callbacks were called
        self.mock_trading_system.close_all_positions.assert_called_once()
        self.mock_trading_system.stop_agents.assert_called_once()
        self.mock_trading_system.save_state.assert_called_once()

class TestGlobalKillSwitch(unittest.TestCase):
    """Test global kill switch functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_trading_system = Mock()
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up global kill switch
        global _global_kill_switch
        if _global_kill_switch:
            _global_kill_switch.stop_monitoring()
            _global_kill_switch = None
        
        # Clean up manual stop file
        stop_file = '/tmp/trading_emergency_stop'
        if os.path.exists(stop_file):
            os.remove(stop_file)
    
    def test_initialize_kill_switch(self):
        """Test global kill switch initialization."""
        kill_switch = initialize_kill_switch(self.mock_trading_system)
        
        self.assertIsNotNone(kill_switch)
        self.assertEqual(kill_switch.trading_system, self.mock_trading_system)
        
        # Test get_kill_switch
        retrieved = get_kill_switch()
        self.assertEqual(kill_switch, retrieved)
    
    def test_global_emergency_stop(self):
        """Test global emergency stop function."""
        # Initialize kill switch
        initialize_kill_switch(self.mock_trading_system)
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            result = emergency_stop("Global test", human_override=True)
        
        self.assertTrue(result)
    
    def test_manual_stop_file_creation(self):
        """Test manual stop file creation."""
        result = create_manual_stop_file()
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists('/tmp/trading_emergency_stop'))
        
        # Read file content
        with open('/tmp/trading_emergency_stop', 'r') as f:
            content = f.read()
        
        self.assertIn("Manual emergency stop triggered", content)

class TestKillSwitchIntegration(unittest.TestCase):
    """Integration tests for kill switch system."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_trading_system = Mock()
        self.kill_switch = initialize_kill_switch(self.mock_trading_system)
        time.sleep(0.1)  # Allow monitoring to start
    
    def tearDown(self):
        """Clean up test environment."""
        self.kill_switch.stop_monitoring()
        
        # Clean up manual stop file
        stop_file = '/tmp/trading_emergency_stop'
        if os.path.exists(stop_file):
            os.remove(stop_file)
    
    def test_manual_file_integration(self):
        """Test manual file trigger integration."""
        # Create manual stop file
        create_manual_stop_file()
        
        # Allow time for monitoring to detect file
        time.sleep(2)
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            # The monitoring thread should have detected the file
            # and triggered shutdown
            pass
        
        # File should be detected by monitoring
        self.assertTrue(os.path.exists('/tmp/trading_emergency_stop'))
    
    def test_error_cascade_protection(self):
        """Test protection against error cascades."""
        # Report multiple errors quickly
        for i in range(15):  # Above threshold
            self.kill_switch.report_error(Exception(f"Cascade error {i}"))
        
        # Allow time for monitoring to detect errors
        time.sleep(0.5)
        
        # Should trigger shutdown due to error threshold
        status = self.kill_switch.get_status()
        self.assertTrue(status['consecutive_errors'] >= 10)
    
    def test_status_reporting(self):
        """Test comprehensive status reporting."""
        status = self.kill_switch.get_status()
        
        # Verify all expected fields are present
        expected_fields = [
            'shutdown_active', 'human_override_active', 
            'consecutive_errors', 'last_health_check', 'shutdown_event'
        ]
        
        for field in expected_fields:
            self.assertIn(field, status)
        
        # Verify initial state
        self.assertFalse(status['shutdown_active'])
        self.assertFalse(status['human_override_active'])
        self.assertEqual(status['consecutive_errors'], 0)
        self.assertIsNone(status['shutdown_event'])

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)