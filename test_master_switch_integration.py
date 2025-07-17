#!/usr/bin/env python3
"""
Test script to verify the master switch integration in execution layer components.

This script tests:
1. Normal operation when system is active
2. Blocked operation when kill switch is active
3. Blocked operation when emergency stop is active
4. Blocked operation when maintenance mode is active
"""

import asyncio
import logging
import sys
import os
import time
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
from src.core.events import EventBus
from src.operations.operational_controls import OperationalControls
from src.safety.kill_switch import initialize_kill_switch, get_kill_switch
from src.execution.execution_engine import ExecutionEngine
from src.components.live_execution_handler import LiveExecutionHandler
from src.components.execution_handler import BaseExecutionHandler, LiveExecutionHandler as ComponentLiveHandler, BacktestExecutionHandler


class TestMasterSwitchIntegration:
    """Test master switch integration in execution components."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.operational_controls = OperationalControls(self.event_bus)
        self.kill_switch = initialize_kill_switch()
        
        # Initialize components with safety controls
        self.execution_engine = ExecutionEngine(
            operational_controls=self.operational_controls
        )
        
        # Test configurations
        self.test_config = {
            "symbol": "NQ",
            "execution_handler": {
                "broker": "interactive_brokers"
            },
            "risk_management": {
                "daily_loss_limit": 5000,
                "max_drawdown": 0.15
            }
        }
        
        logger.info("Test framework initialized")
    
    async def test_execution_engine_normal_operation(self):
        """Test ExecutionEngine normal operation when system is active."""
        logger.info("Testing ExecutionEngine normal operation...")
        
        try:
            # Initialize the engine
            await self.execution_engine.initialize()
            
            # Test processing an execution event
            test_event = {
                'correlation_id': 'test_001',
                'action': 'buy',
                'confidence': 0.85,
                'execution_command': {
                    'action': 'execute_trade',
                    'symbol': 'NQ',
                    'side': 'BUY',
                    'quantity': 1
                }
            }
            
            # This should work normally
            await self.execution_engine._process_execution_event(test_event)
            
            # Get stats
            stats = self.execution_engine.get_stats()
            logger.info(f"ExecutionEngine stats: {stats}")
            
            assert stats['system_state'] == 'active'
            assert stats['executions_processed'] > 0
            
            logger.info("✅ ExecutionEngine normal operation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ExecutionEngine normal operation test failed: {e}")
            return False
    
    async def test_execution_engine_kill_switch_blocked(self):
        """Test ExecutionEngine blocked operation when kill switch is active."""
        logger.info("Testing ExecutionEngine with kill switch active...")
        
        try:
            # Activate kill switch
            self.kill_switch.emergency_stop("Test kill switch")
            
            # Test processing an execution event
            test_event = {
                'correlation_id': 'test_002',
                'action': 'buy',
                'confidence': 0.85,
                'execution_command': {
                    'action': 'execute_trade',
                    'symbol': 'NQ',
                    'side': 'BUY',
                    'quantity': 1
                }
            }
            
            # This should be blocked and not execute
            initial_processed = self.execution_engine.executions_processed
            await self.execution_engine._process_execution_event(test_event)
            
            # Check that no new executions were processed
            assert self.execution_engine.executions_processed == initial_processed
            
            # Get stats
            stats = self.execution_engine.get_stats()
            logger.info(f"ExecutionEngine stats with kill switch: {stats}")
            
            assert stats['system_state'] == 'kill_switch_active'
            
            logger.info("✅ ExecutionEngine kill switch blocked test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ExecutionEngine kill switch blocked test failed: {e}")
            return False
    
    async def test_execution_engine_emergency_stop_blocked(self):
        """Test ExecutionEngine blocked operation when emergency stop is active."""
        logger.info("Testing ExecutionEngine with emergency stop active...")
        
        try:
            # Reset kill switch but activate emergency stop
            self.kill_switch.kill_switch_core.shutdown_active = False
            self.operational_controls.trigger_emergency_stop()
            
            # Test processing an execution event
            test_event = {
                'correlation_id': 'test_003',
                'action': 'buy',
                'confidence': 0.85,
                'execution_command': {
                    'action': 'execute_trade',
                    'symbol': 'NQ',
                    'side': 'BUY',
                    'quantity': 1
                }
            }
            
            # This should be blocked and not execute
            initial_processed = self.execution_engine.executions_processed
            await self.execution_engine._process_execution_event(test_event)
            
            # Check that no new executions were processed
            assert self.execution_engine.executions_processed == initial_processed
            
            # Get stats
            stats = self.execution_engine.get_stats()
            logger.info(f"ExecutionEngine stats with emergency stop: {stats}")
            
            assert stats['system_state'] == 'emergency_stop'
            
            logger.info("✅ ExecutionEngine emergency stop blocked test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ExecutionEngine emergency stop blocked test failed: {e}")
            return False
    
    async def test_execution_engine_maintenance_mode_blocked(self):
        """Test ExecutionEngine blocked operation when maintenance mode is active."""
        logger.info("Testing ExecutionEngine with maintenance mode active...")
        
        try:
            # Reset emergency stop but activate maintenance mode
            self.operational_controls.reset_emergency_stop()
            self.operational_controls.enable_maintenance_mode()
            
            # Test processing an execution event
            test_event = {
                'correlation_id': 'test_004',
                'action': 'buy',
                'confidence': 0.85,
                'execution_command': {
                    'action': 'execute_trade',
                    'symbol': 'NQ',
                    'side': 'BUY',
                    'quantity': 1
                }
            }
            
            # This should be blocked and not execute
            initial_processed = self.execution_engine.executions_processed
            await self.execution_engine._process_execution_event(test_event)
            
            # Check that no new executions were processed
            assert self.execution_engine.executions_processed == initial_processed
            
            # Get stats
            stats = self.execution_engine.get_stats()
            logger.info(f"ExecutionEngine stats with maintenance mode: {stats}")
            
            assert stats['system_state'] == 'maintenance_mode'
            
            logger.info("✅ ExecutionEngine maintenance mode blocked test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ExecutionEngine maintenance mode blocked test failed: {e}")
            return False
    
    async def test_live_execution_handler_normal_operation(self):
        """Test LiveExecutionHandler normal operation when system is active."""
        logger.info("Testing LiveExecutionHandler normal operation...")
        
        try:
            # Reset all controls to normal state
            self.operational_controls.disable_maintenance_mode()
            self.operational_controls.reset_emergency_stop()
            
            # Create LiveExecutionHandler
            live_handler = LiveExecutionHandler(
                self.test_config,
                self.event_bus,
                self.operational_controls
            )
            
            # Initialize it
            await live_handler.initialize()
            
            # Test executing a trade
            test_trade = {
                'action': 'BUY',
                'quantity': 1,
                'price': 18000.0,
                'order_type': 'LIMIT'
            }
            
            # This should work normally
            await live_handler.execute_trade(test_trade)
            
            # Get status
            status = live_handler.get_status()
            logger.info(f"LiveExecutionHandler status: {status}")
            
            assert status['system_state'] == 'active'
            
            logger.info("✅ LiveExecutionHandler normal operation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ LiveExecutionHandler normal operation test failed: {e}")
            return False
    
    async def test_live_execution_handler_blocked_operation(self):
        """Test LiveExecutionHandler blocked operation when emergency stop is active."""
        logger.info("Testing LiveExecutionHandler with emergency stop active...")
        
        try:
            # Activate emergency stop
            self.operational_controls.trigger_emergency_stop()
            
            # Create LiveExecutionHandler
            live_handler = LiveExecutionHandler(
                self.test_config,
                self.event_bus,
                self.operational_controls
            )
            
            # Initialize it
            await live_handler.initialize()
            
            # Test executing a trade
            test_trade = {
                'action': 'BUY',
                'quantity': 1,
                'price': 18000.0,
                'order_type': 'LIMIT'
            }
            
            # This should be blocked and not execute
            initial_orders = live_handler.total_orders
            await live_handler.execute_trade(test_trade)
            
            # Check that no new orders were created
            assert live_handler.total_orders == initial_orders
            
            # Get status
            status = live_handler.get_status()
            logger.info(f"LiveExecutionHandler status with emergency stop: {status}")
            
            assert status['system_state'] == 'emergency_stop'
            
            logger.info("✅ LiveExecutionHandler blocked operation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ LiveExecutionHandler blocked operation test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting master switch integration tests...")
        
        tests = [
            self.test_execution_engine_normal_operation,
            self.test_execution_engine_kill_switch_blocked,
            self.test_execution_engine_emergency_stop_blocked,
            self.test_execution_engine_maintenance_mode_blocked,
            self.test_live_execution_handler_normal_operation,
            self.test_live_execution_handler_blocked_operation
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
    
    test_runner = TestMasterSwitchIntegration()
    success = await test_runner.run_all_tests()
    
    if success:
        logger.info("✅ Master switch integration is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Master switch integration has issues!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())