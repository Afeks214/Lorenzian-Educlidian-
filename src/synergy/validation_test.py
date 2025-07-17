#!/usr/bin/env python3
"""
Sequential Synergy System Validation Test
=========================================

This script validates the sequential synergy detection system to ensure:
- NW-RQK → MLMI → FVG chain processing works correctly
- Signal handoffs are properly coordinated
- State management functions correctly
- Integration bridge handles handoffs properly
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.synergy.detector import SynergyDetector
from src.synergy.base import Signal, SynergyPattern
from src.synergy.state_manager import SynergyStateManager
from src.synergy.integration_bridge import SynergyIntegrationBridge
from src.core.minimal_dependencies import AlgoSpaceKernel, EventType, Event


class MockKernel:
    """Mock kernel for testing."""
    
    def __init__(self):
        self.config = {
            'synergy_detector': {
                'time_window_bars': 10,
                'cooldown_bars': 5,
                'synergy_expiration_minutes': 30,
                'mlmi_threshold': 0.5,
                'nwrqk_threshold': 0.3,
                'fvg_min_size': 0.001
            }
        }
        self.event_bus = MockEventBus()


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
        self.subscribers = {}
    
    def subscribe(self, event_type, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type, handler):
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    def create_event(self, event_type, payload, source):
        return Event(event_type, payload, datetime.now(), source)
    
    def publish(self, event):
        self.events.append(event)
    
    async def publish(self, event):
        self.events.append(event)


class SequentialSynergyValidationTest:
    """Test suite for sequential synergy system."""
    
    def __init__(self):
        self.kernel = MockKernel()
        self.detector = None
        self.test_results = []
    
    async def setup(self):
        """Setup test environment."""
        self.detector = SynergyDetector("test_detector", self.kernel)
        await self.detector.initialize()
        logger.info("Test setup completed")
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.detector:
            await self.detector.shutdown()
        logger.info("Test teardown completed")
    
    def create_test_features(self, 
                           nwrqk_signal: int = 0, 
                           nwrqk_slope: float = 0.0,
                           mlmi_signal: int = 0, 
                           mlmi_value: float = 50.0,
                           fvg_mitigation: bool = False,
                           fvg_bullish_mitigated: bool = False,
                           fvg_bearish_mitigated: bool = False,
                           timestamp: datetime = None) -> Dict[str, Any]:
        """Create test features for synergy detection."""
        if timestamp is None:
            timestamp = datetime.now()
        
        return {
            'timestamp': timestamp,
            'current_price': 100.0,
            'volatility_30': 0.02,
            'volume_ratio': 1.2,
            'volume_momentum_30': 0.1,
            'lvn_nearest_price': 99.5,
            'lvn_nearest_strength': 0.8,
            'lvn_distance_points': 0.5,
            
            # NW-RQK features
            'nwrqk_signal': nwrqk_signal,
            'nwrqk_slope': nwrqk_slope,
            'nwrqk_value': 100.0,
            
            # MLMI features
            'mlmi_signal': mlmi_signal,
            'mlmi_value': mlmi_value,
            
            # FVG features
            'fvg_mitigation_signal': fvg_mitigation,
            'fvg_bullish_mitigated': fvg_bullish_mitigated,
            'fvg_bearish_mitigated': fvg_bearish_mitigated,
            'fvg_bullish_size': 0.002 if fvg_bullish_mitigated else 0.0,
            'fvg_bearish_size': 0.002 if fvg_bearish_mitigated else 0.0,
            'fvg_bullish_level': 100.2 if fvg_bullish_mitigated else 100.0,
            'fvg_bearish_level': 99.8 if fvg_bearish_mitigated else 100.0
        }
    
    async def test_sequential_chain_processing(self):
        """Test that signals are processed in NW-RQK → MLMI → FVG sequence."""
        logger.info("Testing sequential chain processing...")
        
        # Test Case 1: Only NW-RQK signal (should not create synergy)
        features = self.create_test_features(nwrqk_signal=1, nwrqk_slope=0.5)
        synergy = self.detector.process_features(features, features['timestamp'])\n        \n        if synergy is None:\n            self.test_results.append((\"Sequential Step 1\", \"PASS\", \"NW-RQK alone does not create synergy\"))\n        else:\n            self.test_results.append((\"Sequential Step 1\", \"FAIL\", \"NW-RQK alone should not create synergy\"))\n        \n        # Test Case 2: NW-RQK + MLMI (should not create synergy)\n        features = self.create_test_features(\n            nwrqk_signal=1, nwrqk_slope=0.5,\n            mlmi_signal=1, mlmi_value=75.0\n        )\n        synergy = self.detector.process_features(features, features['timestamp'])\n        \n        if synergy is None:\n            self.test_results.append((\"Sequential Step 2\", \"PASS\", \"NW-RQK + MLMI does not create synergy\"))\n        else:\n            self.test_results.append((\"Sequential Step 2\", \"FAIL\", \"NW-RQK + MLMI should not create synergy\"))\n        \n        # Test Case 3: Full sequential chain (should create synergy)\n        features = self.create_test_features(\n            nwrqk_signal=1, nwrqk_slope=0.5,\n            mlmi_signal=1, mlmi_value=75.0,\n            fvg_mitigation=True, fvg_bullish_mitigated=True\n        )\n        synergy = self.detector.process_features(features, features['timestamp'])\n        \n        if synergy and synergy.synergy_type == 'SEQUENTIAL_SYNERGY':\n            self.test_results.append((\"Sequential Step 3\", \"PASS\", \"Full chain creates SEQUENTIAL_SYNERGY\"))\n        else:\n            self.test_results.append((\"Sequential Step 3\", \"FAIL\", f\"Expected SEQUENTIAL_SYNERGY, got {synergy.synergy_type if synergy else None}\"))\n        \n        logger.info(\"Sequential chain processing test completed\")\n    \n    async def test_signal_validation(self):\n        \"\"\"Test signal chain validation.\"\"\"\n        logger.info(\"Testing signal validation...\")\n        \n        now = datetime.now()\n        \n        # Create test signals\n        nwrqk_signal = Signal('nwrqk', 1, now, 100.0, 0.8)\n        mlmi_signal_valid = Signal('mlmi', 1, now + timedelta(minutes=1), 100.0, 0.7)\n        mlmi_signal_invalid = Signal('mlmi', -1, now + timedelta(minutes=1), 100.0, 0.7)\n        fvg_signal = Signal('fvg', 1, now + timedelta(minutes=2), 100.0, 0.6)\n        \n        # Test valid chain\n        valid_chain = self.detector._validate_signal_chain(nwrqk_signal, mlmi_signal_valid)\n        if valid_chain:\n            self.test_results.append((\"Signal Validation\", \"PASS\", \"Valid signal chain accepted\"))\n        else:\n            self.test_results.append((\"Signal Validation\", \"FAIL\", \"Valid signal chain rejected\"))\n        \n        # Test invalid chain (direction mismatch)\n        invalid_chain = self.detector._validate_signal_chain(nwrqk_signal, mlmi_signal_invalid)\n        if not invalid_chain:\n            self.test_results.append((\"Signal Validation\", \"PASS\", \"Invalid signal chain rejected\"))\n        else:\n            self.test_results.append((\"Signal Validation\", \"FAIL\", \"Invalid signal chain accepted\"))\n        \n        logger.info(\"Signal validation test completed\")\n    \n    async def test_state_management(self):\n        \"\"\"Test synergy state management.\"\"\"\n        logger.info(\"Testing state management...\")\n        \n        # Create a test synergy\n        now = datetime.now()\n        signals = [\n            Signal('nwrqk', 1, now, 100.0, 0.8),\n            Signal('mlmi', 1, now + timedelta(minutes=1), 100.0, 0.7),\n            Signal('fvg', 1, now + timedelta(minutes=2), 100.0, 0.6)\n        ]\n        \n        synergy = SynergyPattern(\n            synergy_type='SEQUENTIAL_SYNERGY',\n            direction=1,\n            signals=signals,\n            completion_time=now + timedelta(minutes=2),\n            bars_to_complete=3\n        )\n        \n        # Test state record creation\n        synergy_id = self.detector.state_manager.create_synergy_record(synergy)\n        if synergy_id:\n            self.test_results.append((\"State Management\", \"PASS\", \"State record created successfully\"))\n        else:\n            self.test_results.append((\"State Management\", \"FAIL\", \"State record creation failed\"))\n        \n        # Test state retrieval\n        state_record = self.detector.state_manager.get_synergy_state(synergy_id)\n        if state_record and state_record.synergy_pattern.synergy_type == 'SEQUENTIAL_SYNERGY':\n            self.test_results.append((\"State Management\", \"PASS\", \"State record retrieved successfully\"))\n        else:\n            self.test_results.append((\"State Management\", \"FAIL\", \"State record retrieval failed\"))\n        \n        logger.info(\"State management test completed\")\n    \n    async def test_integration_handoff(self):\n        \"\"\"Test integration handoff functionality.\"\"\"\n        logger.info(\"Testing integration handoff...\")\n        \n        # Create a test synergy with state management\n        now = datetime.now()\n        signals = [\n            Signal('nwrqk', 1, now, 100.0, 0.8),\n            Signal('mlmi', 1, now + timedelta(minutes=1), 100.0, 0.7),\n            Signal('fvg', 1, now + timedelta(minutes=2), 100.0, 0.6)\n        ]\n        \n        synergy = SynergyPattern(\n            synergy_type='SEQUENTIAL_SYNERGY',\n            direction=1,\n            signals=signals,\n            completion_time=now + timedelta(minutes=2),\n            bars_to_complete=3\n        )\n        \n        # Create state record\n        synergy_id = self.detector.state_manager.create_synergy_record(synergy)\n        synergy.synergy_id = synergy_id\n        synergy.state_managed = True\n        \n        # Test handoff\n        handoff_success = await self.detector.integration_bridge.initiate_handoff(\n            synergy, 'execution_system'\n        )\n        \n        if handoff_success:\n            self.test_results.append((\"Integration Handoff\", \"PASS\", \"Handoff to execution system successful\"))\n        else:\n            self.test_results.append((\"Integration Handoff\", \"FAIL\", \"Handoff to execution system failed\"))\n        \n        logger.info(\"Integration handoff test completed\")\n    \n    async def test_end_to_end_workflow(self):\n        \"\"\"Test complete end-to-end workflow.\"\"\"\n        logger.info(\"Testing end-to-end workflow...\")\n        \n        # Simulate sequential signal detection\n        base_time = datetime.now()\n        \n        # Step 1: NW-RQK signal\n        features_1 = self.create_test_features(\n            nwrqk_signal=1, nwrqk_slope=0.5,\n            timestamp=base_time\n        )\n        synergy_1 = self.detector.process_features(features_1, base_time)\n        \n        # Step 2: MLMI signal (after NW-RQK)\n        features_2 = self.create_test_features(\n            nwrqk_signal=1, nwrqk_slope=0.5,\n            mlmi_signal=1, mlmi_value=75.0,\n            timestamp=base_time + timedelta(minutes=1)\n        )\n        synergy_2 = self.detector.process_features(features_2, base_time + timedelta(minutes=1))\n        \n        # Step 3: FVG signal (completing the chain)\n        features_3 = self.create_test_features(\n            nwrqk_signal=1, nwrqk_slope=0.5,\n            mlmi_signal=1, mlmi_value=75.0,\n            fvg_mitigation=True, fvg_bullish_mitigated=True,\n            timestamp=base_time + timedelta(minutes=2)\n        )\n        synergy_3 = self.detector.process_features(features_3, base_time + timedelta(minutes=2))\n        \n        # Check that only the final step created a synergy\n        if synergy_1 is None and synergy_2 is None and synergy_3 is not None:\n            self.test_results.append((\"End-to-End\", \"PASS\", \"Sequential processing works correctly\"))\n        else:\n            self.test_results.append((\"End-to-End\", \"FAIL\", f\"Unexpected synergy creation: {synergy_1 is not None}, {synergy_2 is not None}, {synergy_3 is not None}\"))\n        \n        # Check synergy properties\n        if synergy_3 and synergy_3.is_sequential():\n            self.test_results.append((\"End-to-End\", \"PASS\", \"Generated sequential synergy\"))\n        else:\n            self.test_results.append((\"End-to-End\", \"FAIL\", \"Did not generate sequential synergy\"))\n        \n        logger.info(\"End-to-end workflow test completed\")\n    \n    def print_results(self):\n        \"\"\"Print test results.\"\"\"\n        logger.info(\"\\n\" + \"=\"*80)\n        logger.info(\"SEQUENTIAL SYNERGY SYSTEM VALIDATION RESULTS\")\n        logger.info(\"=\"*80)\n        \n        passed = 0\n        failed = 0\n        \n        for test_name, result, description in self.test_results:\n            status_symbol = \"✓\" if result == \"PASS\" else \"✗\"\n            logger.info(f\"{status_symbol} {test_name:<20} {result:<6} {description}\")\n            \n            if result == \"PASS\":\n                passed += 1\n            else:\n                failed += 1\n        \n        logger.info(\"=\"*80)\n        logger.info(f\"SUMMARY: {passed} passed, {failed} failed\")\n        \n        if failed == 0:\n            logger.info(\"🎉 ALL TESTS PASSED - Sequential synergy system is working correctly!\")\n        else:\n            logger.error(f\"❌ {failed} TESTS FAILED - Sequential synergy system needs fixes\")\n        \n        logger.info(\"=\"*80)\n    \n    async def run_all_tests(self):\n        \"\"\"Run all validation tests.\"\"\"\n        try:\n            await self.setup()\n            \n            await self.test_sequential_chain_processing()\n            await self.test_signal_validation()\n            await self.test_state_management()\n            await self.test_integration_handoff()\n            await self.test_end_to_end_workflow()\n            \n            self.print_results()\n            \n        except Exception as e:\n            logger.error(f\"Test execution failed: {e}\")\n            raise\n        finally:\n            await self.teardown()\n\n\nasync def main():\n    \"\"\"Main test runner.\"\"\"\n    logger.info(\"Starting Sequential Synergy System Validation\")\n    \n    validator = SequentialSynergyValidationTest()\n    await validator.run_all_tests()\n    \n    logger.info(\"Validation completed\")\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n"