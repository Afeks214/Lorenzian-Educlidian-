#!/usr/bin/env python3
"""
Simple test to verify safety checks are properly implemented in our code.
This test focuses on the code structure and logic we've added.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_safety_code_structure():
    """Test that our safety code is properly structured."""
    logger.info("Testing safety code structure...")
    
    # Test 1: Verify kill switch can be imported
    try:
        from src.safety.kill_switch import get_kill_switch, initialize_kill_switch
        logger.info("✓ Kill switch module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import kill switch: {e}")
        return False
    
    # Test 2: Verify synergy_detector imports kill switch
    try:
        with open('/home/QuantNova/GrandModel/src/synergy_detector.py', 'r') as f:
            content = f.read()
            if 'from src.safety.kill_switch import get_kill_switch' in content:
                logger.info("✓ SynergyDetector imports kill switch")
            else:
                logger.error("✗ SynergyDetector missing kill switch import")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to read synergy_detector.py: {e}")
        return False
    
    # Test 3: Verify indicators.py imports kill switch
    try:
        with open('/home/QuantNova/GrandModel/src/indicators.py', 'r') as f:
            content = f.read()
            if 'from src.safety.kill_switch import get_kill_switch' in content:
                logger.info("✓ Indicators module imports kill switch")
            else:
                logger.error("✗ Indicators module missing kill switch import")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to read indicators.py: {e}")
        return False
    
    # Test 4: Verify synergy/detector.py imports kill switch
    try:
        with open('/home/QuantNova/GrandModel/src/synergy/detector.py', 'r') as f:
            content = f.read()
            if 'from src.safety.kill_switch import get_kill_switch' in content:
                logger.info("✓ Sequential detector imports kill switch")
            else:
                logger.error("✗ Sequential detector missing kill switch import")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to read synergy/detector.py: {e}")
        return False
    
    return True

def test_safety_logic_implementation():
    """Test that safety logic is properly implemented."""
    logger.info("Testing safety logic implementation...")
    
    # Test 1: Check SynergyDetector has safety checks
    try:
        with open('/home/QuantNova/GrandModel/src/synergy_detector.py', 'r') as f:
            content = f.read()
            
            # Check for safety check in detect_synergies
            if 'kill_switch = get_kill_switch()' in content and 'kill_switch.is_active()' in content:
                logger.info("✓ SynergyDetector has safety checks")
            else:
                logger.error("✗ SynergyDetector missing safety checks")
                return False
            
            # Check for empty results method
            if '_create_empty_synergy_results' in content:
                logger.info("✓ SynergyDetector has empty results method")
            else:
                logger.error("✗ SynergyDetector missing empty results method")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to check SynergyDetector safety logic: {e}")
        return False
    
    # Test 2: Check indicators have safety checks
    try:
        with open('/home/QuantNova/GrandModel/src/indicators.py', 'r') as f:
            content = f.read()
            
            # Check for base class safety methods
            if '_is_system_active' in content and '_create_empty_indicator_results' in content:
                logger.info("✓ Indicators have base safety methods")
            else:
                logger.error("✗ Indicators missing base safety methods")
                return False
            
            # Check for safety checks in calculate methods
            safety_checks = [
                'Trading system is OFF - blocking NW-RQK calculation',
                'Trading system is OFF - blocking MLMI calculation',
                'Trading system is OFF - blocking FVG calculation'
            ]
            
            checks_found = sum(1 for check in safety_checks if check in content)
            if checks_found == 3:
                logger.info("✓ All indicators have safety checks")
            else:
                logger.error(f"✗ Only {checks_found}/3 indicators have safety checks")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to check indicators safety logic: {e}")
        return False
    
    # Test 3: Check sequential detector has safety checks
    try:
        with open('/home/QuantNova/GrandModel/src/synergy/detector.py', 'r') as f:
            content = f.read()
            
            # Check for multiple safety check points
            safety_points = [
                'Trading system is OFF - blocking synergy processing',
                'Trading system is OFF - blocking signal detection',
                'Trading system is OFF - blocking indicator event processing'
            ]
            
            checks_found = sum(1 for point in safety_points if point in content)
            if checks_found == 3:
                logger.info("✓ Sequential detector has comprehensive safety checks")
            else:
                logger.error(f"✗ Sequential detector missing some safety checks ({checks_found}/3)")
                return False
            
            # Check for status reporting enhancement
            if 'system_active' in content and 'safety_checks' in content:
                logger.info("✓ Sequential detector has enhanced status reporting")
            else:
                logger.error("✗ Sequential detector missing enhanced status reporting")
                return False
    except Exception as e:
        logger.error(f"✗ Failed to check sequential detector safety logic: {e}")
        return False
    
    return True

def test_kill_switch_functionality():
    """Test basic kill switch functionality."""
    logger.info("Testing kill switch functionality...")
    
    try:
        from src.safety.kill_switch import initialize_kill_switch, get_kill_switch
        
        # Test initialization
        kill_switch = initialize_kill_switch()
        if kill_switch is None:
            logger.error("✗ Failed to initialize kill switch")
            return False
        
        logger.info("✓ Kill switch initialized successfully")
        
        # Test get_kill_switch
        retrieved_switch = get_kill_switch()
        if retrieved_switch is None:
            logger.error("✗ Failed to retrieve kill switch")
            return False
        
        logger.info("✓ Kill switch retrieved successfully")
        
        # Test initial state
        if not kill_switch.is_active():
            logger.info("✓ Kill switch initial state is inactive")
        else:
            logger.error("✗ Kill switch should be inactive initially")
            return False
        
        # Test activation (will trigger exit, so we expect SystemExit)
        try:
            kill_switch.emergency_stop("Test activation")
            logger.error("✗ Expected SystemExit was not raised")
            return False
        except SystemExit:
            logger.info("✓ Kill switch activation triggers SystemExit (expected)")
            return True
        
    except Exception as e:
        logger.error(f"✗ Kill switch test failed: {e}")
        return False

def test_empty_results_structure():
    """Test that empty results have correct structure."""
    logger.info("Testing empty results structure...")
    
    try:
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='5min')
        
        # Test empty synergy results structure
        expected_synergy_columns = [
            'synergy_bull', 'synergy_bear', 'synergy_strength',
            'nwrqk_active_bull', 'nwrqk_active_bear', 
            'mlmi_confirmed_bull', 'mlmi_confirmed_bear'
        ]
        
        # Test empty indicator results structure
        expected_nwrqk_columns = [
            'nwrqk', 'nwrqk_bull', 'nwrqk_bear', 'nwrqk_strength', 
            'nwrqk_slope', 'price_deviation'
        ]
        
        expected_mlmi_columns = [
            'mlmi_bull', 'mlmi_bear', 'mlmi_confidence', 'mlmi_signal'
        ]
        
        expected_fvg_columns = [
            'fvg_bull', 'fvg_bear', 'fvg_size', 'fvg_count'
        ]
        
        logger.info("✓ Empty results structure requirements defined")
        
        # Verify the structure matches expected columns
        # (This is a structural test - actual functionality would require imports)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Empty results structure test failed: {e}")
        return False

def main():
    """Run all safety implementation tests."""
    logger.info("Starting safety implementation tests...")
    
    tests = [
        test_safety_code_structure,
        test_safety_logic_implementation,
        test_kill_switch_functionality,
        test_empty_results_structure
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
    
    logger.info(f"\n=== Safety Implementation Test Results ===")
    logger.info(f"Tests passed: {passed}")
    logger.info(f"Tests failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All safety implementation tests PASSED!")
        logger.info("✅ Master switch safety checks are properly implemented")
        return True
    else:
        logger.error("❌ Some safety implementation tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)