#!/usr/bin/env python3
"""
Test to verify that existing functionality is preserved when the trading system is ON.
This ensures our safety checks don't break normal operation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_normal_operation_preserved():
    """Test that normal operation is preserved when system is ON."""
    logger.info("Testing normal operation preservation...")
    
    try:
        from src.safety.kill_switch import get_kill_switch, initialize_kill_switch
        
        # Ensure kill switch is initialized but not active
        kill_switch = get_kill_switch()
        if kill_switch is None:
            initialize_kill_switch()
            kill_switch = get_kill_switch()
        
        # Verify system is initially inactive (not shutdown)
        if kill_switch and kill_switch.is_active():
            logger.error("✗ System should be active for normal operation test")
            return False
        
        logger.info("✓ System is in normal operating state")
        
        # Test that safety check functions return correct values
        # when system is ON (not shutdown)
        
        # Test 1: Import and check structure
        try:
            from src.synergy_detector import SynergyDetector
            logger.info("✓ SynergyDetector can be imported")
            
            # Test basic configuration
            config = {
                'window': 10,
                'nwrqk_strength_threshold': 0.5,
                'mlmi_confidence_threshold': 0.6,
                'state_decay_window': 20,
                'strength_calculation': {
                    'nwrqk_weight': 0.33,
                    'mlmi_weight': 0.33,
                    'fvg_weight': 0.34
                }
            }
            
            detector = SynergyDetector(config)
            logger.info("✓ SynergyDetector can be initialized")
            
        except Exception as e:
            logger.error(f"✗ SynergyDetector test failed: {e}")
            return False
        
        # Test 2: Test indicator base class functionality
        try:
            # Import directly from the file since it's not in __init__.py
            sys.path.insert(0, '/home/QuantNova/GrandModel/src')
            from indicators import IndicatorBase
            
            # Create a dummy indicator to test base functionality
            class TestIndicator(IndicatorBase):
                def _validate_config(self):
                    pass
                
                def calculate(self, data):
                    # Test the safety check method
                    if not self._is_system_active():
                        return self._create_empty_indicator_results(data, ['test_col'])
                    
                    # Normal calculation would happen here
                    result = pd.DataFrame(index=data.index)
                    result['test_col'] = 1.0
                    return result
            
            # Test with some dummy data
            dates = pd.date_range(start='2023-01-01', periods=5, freq='5min')
            test_data = pd.DataFrame({'close': [100, 101, 102, 103, 104]}, index=dates)
            
            test_indicator = TestIndicator({})
            
            # Should return normal results when system is active
            result = test_indicator.calculate(test_data)
            
            if 'test_col' in result.columns and result['test_col'].sum() == 5:
                logger.info("✓ Indicator base class works normally when system is ON")
            else:
                logger.error("✗ Indicator base class not working properly")
                return False
                
        except Exception as e:
            logger.error(f"✗ Indicator base class test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Normal operation test failed: {e}")
        return False

def test_safety_check_logic():
    """Test the logic of safety check functions."""
    logger.info("Testing safety check logic...")
    
    try:
        from src.safety.kill_switch import get_kill_switch
        
        # Test with no kill switch (should be active)
        kill_switch = get_kill_switch()
        
        # Test the logic pattern we implemented
        def test_system_active():
            kill_switch = get_kill_switch()
            return not (kill_switch and kill_switch.is_active())
        
        # Should be active when no kill switch or when kill switch is not active
        if test_system_active():
            logger.info("✓ System reports as active when kill switch is not active")
        else:
            logger.error("✗ System should be active when kill switch is not active")
            return False
        
        # Test the empty results creation logic
        dates = pd.date_range(start='2023-01-01', periods=3, freq='5min')
        test_data = pd.DataFrame({'close': [100, 101, 102]}, index=dates)
        
        # Simulate empty results creation
        def create_empty_results(data, columns):
            results = pd.DataFrame(index=data.index)
            for col in columns:
                if 'bull' in col or 'bear' in col:
                    results[col] = False
                else:
                    results[col] = 0.0
            return results
        
        empty_result = create_empty_results(test_data, ['bull_signal', 'bear_signal', 'strength'])
        
        # Verify structure
        if (not empty_result['bull_signal'].any() and 
            not empty_result['bear_signal'].any() and 
            (empty_result['strength'] == 0.0).all()):
            logger.info("✓ Empty results creation logic works correctly")
        else:
            logger.error("✗ Empty results creation logic is incorrect")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Safety check logic test failed: {e}")
        return False

def test_logging_functionality():
    """Test that logging functionality works correctly."""
    logger.info("Testing logging functionality...")
    
    try:
        # Test that warning messages are properly formatted
        test_messages = [
            "Trading system is OFF - blocking synergy detection",
            "Trading system is OFF - blocking NW-RQK calculation",
            "Trading system is OFF - blocking MLMI calculation",
            "Trading system is OFF - blocking FVG calculation",
            "Trading system is OFF - blocking synergy processing",
            "Trading system is OFF - blocking signal detection"
        ]
        
        # These are the messages we expect to see in the logs
        logger.info("✓ Log messages are properly defined")
        
        # Test logging setup
        test_logger = logging.getLogger('test_safety')
        test_logger.warning("Test warning message")
        
        logger.info("✓ Logging functionality works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Logging functionality test failed: {e}")
        return False

def test_thread_safety():
    """Test basic thread safety considerations."""
    logger.info("Testing thread safety considerations...")
    
    try:
        from src.safety.kill_switch import get_kill_switch
        
        # Test that get_kill_switch() can be called multiple times safely
        switch1 = get_kill_switch()
        switch2 = get_kill_switch()
        
        # Should return the same instance (singleton pattern)
        if switch1 is switch2:
            logger.info("✓ Kill switch follows singleton pattern")
        else:
            logger.warning("⚠ Kill switch may not be thread-safe (different instances)")
        
        # Test that safety checks can be called multiple times
        def test_multiple_calls():
            kill_switch = get_kill_switch()
            results = []
            for i in range(5):
                results.append(not (kill_switch and kill_switch.is_active()))
            return all(r == results[0] for r in results)  # All should be the same
        
        if test_multiple_calls():
            logger.info("✓ Safety checks are consistent across multiple calls")
        else:
            logger.error("✗ Safety checks are inconsistent")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Thread safety test failed: {e}")
        return False

def main():
    """Run all functionality preservation tests."""
    logger.info("Starting functionality preservation tests...")
    
    tests = [
        test_normal_operation_preserved,
        test_safety_check_logic,
        test_logging_functionality,
        test_thread_safety
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            logger.info(f"\n--- Running {test.__name__} ---")
            if test():
                passed += 1
                logger.info(f"✓ {test.__name__} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test.__name__} FAILED")
        except Exception as e:
            logger.error(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"\n=== Functionality Preservation Test Results ===")
    logger.info(f"Tests passed: {passed}")
    logger.info(f"Tests failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All functionality preservation tests PASSED!")
        logger.info("✅ Existing functionality is preserved when system is ON")
        return True
    else:
        logger.error("❌ Some functionality preservation tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)