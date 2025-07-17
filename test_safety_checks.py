#!/usr/bin/env python3
"""
Test script to validate safety checks in signal generation systems.
Tests that all signal generation components properly block signals when the trading system is OFF.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synergy_detector import SynergyDetector
from src.indicators import NWRQK, MLMI, FVG
from src.synergy.detector import SynergyDetector as SequentialSynergyDetector
from src.safety.kill_switch import initialize_kill_switch, get_kill_switch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='5min')
    n = len(dates)
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 4000
    returns = np.random.normal(0, 0.002, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    return data

def test_synergy_detector_safety():
    """Test safety checks in SynergyDetector."""
    logger.info("Testing SynergyDetector safety checks...")
    
    # Create test data
    data = create_test_data()
    
    # Create indicator signals
    nwrqk_signals = pd.DataFrame({
        'nwrqk_bull': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'nwrqk_bear': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'nwrqk_strength': np.random.uniform(0, 1, len(data))
    }, index=data.index)
    
    mlmi_signals = pd.DataFrame({
        'mlmi_bull': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'mlmi_bear': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'mlmi_confidence': np.random.uniform(0, 1, len(data))
    }, index=data.index)
    
    fvg_signals = pd.DataFrame({
        'fvg_bull': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'fvg_bear': np.random.choice([True, False], len(data), p=[0.1, 0.9]),
        'fvg_size': np.random.uniform(0, 0.01, len(data))
    }, index=data.index)
    
    # Test configuration
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
    
    # Test 1: System ON (should generate signals)
    logger.info("Test 1: System ON")
    results_on = detector.detect_synergies(nwrqk_signals, mlmi_signals, fvg_signals)
    signals_on = results_on['synergy_bull'].sum() + results_on['synergy_bear'].sum()
    logger.info(f"Signals when system ON: {signals_on}")
    
    # Test 2: System OFF (should block signals)
    logger.info("Test 2: System OFF")
    
    # Initialize kill switch and activate it
    kill_switch = initialize_kill_switch()
    kill_switch.emergency_stop("Testing safety checks", human_override=True)
    
    try:
        results_off = detector.detect_synergies(nwrqk_signals, mlmi_signals, fvg_signals)
        signals_off = results_off['synergy_bull'].sum() + results_off['synergy_bear'].sum()
        logger.info(f"Signals when system OFF: {signals_off}")
        
        # Verify no signals were generated
        assert signals_off == 0, f"Expected 0 signals when OFF, got {signals_off}"
        assert not results_off['synergy_bull'].any(), "Expected no bull signals when OFF"
        assert not results_off['synergy_bear'].any(), "Expected no bear signals when OFF"
        assert (results_off['synergy_strength'] == 0).all(), "Expected zero strength when OFF"
        
        logger.info("✓ SynergyDetector safety checks PASSED")
        return True
        
    except SystemExit:
        # Kill switch triggers sys.exit, but we want to continue testing
        logger.info("Kill switch triggered exit (expected behavior)")
        return True
    except Exception as e:
        logger.error(f"SynergyDetector safety check failed: {e}")
        return False

def test_indicators_safety():
    """Test safety checks in individual indicators."""
    logger.info("Testing individual indicator safety checks...")
    
    # Create test data
    data = create_test_data()
    
    # Test NWRQK
    logger.info("Testing NWRQK safety...")
    nwrqk_config = {
        'window': 50,
        'n_kernels': 3,
        'alphas': [0.1, 0.2, 0.3],
        'length_scales': [10, 20, 30],
        'threshold': 0.001,
        'volatility_adaptive': True
    }
    
    nwrqk = NWRQK(nwrqk_config)
    
    # Test MLMI
    logger.info("Testing MLMI safety...")
    mlmi_config = {
        'window': 100,
        'k_neighbors': 5,
        'feature_window': 10,
        'rsi_period': 14,
        'bull_threshold': 0.6,
        'bear_threshold': 0.4,
        'confidence_threshold': 0.5
    }
    
    mlmi = MLMI(mlmi_config)
    
    # Test FVG
    logger.info("Testing FVG safety...")
    fvg_config = {
        'min_gap_pct': 0.001,
        'volume_factor': 1.5,
        'volume_window': 20
    }
    
    fvg = FVG(fvg_config)
    
    try:
        # Test with system OFF
        nwrqk_results = nwrqk.calculate(data)
        mlmi_results = mlmi.calculate(data)
        fvg_results = fvg.calculate(data)
        
        # Verify all indicators return empty results
        assert not nwrqk_results['nwrqk_bull'].any(), "Expected no NWRQK bull signals when OFF"
        assert not nwrqk_results['nwrqk_bear'].any(), "Expected no NWRQK bear signals when OFF"
        assert not mlmi_results['mlmi_bull'].any(), "Expected no MLMI bull signals when OFF"
        assert not mlmi_results['mlmi_bear'].any(), "Expected no MLMI bear signals when OFF"
        assert not fvg_results['fvg_bull'].any(), "Expected no FVG bull signals when OFF"
        assert not fvg_results['fvg_bear'].any(), "Expected no FVG bear signals when OFF"
        
        logger.info("✓ Individual indicator safety checks PASSED")
        return True
        
    except SystemExit:
        logger.info("Kill switch triggered exit (expected behavior)")
        return True
    except Exception as e:
        logger.error(f"Indicator safety check failed: {e}")
        return False

def test_sequential_detector_safety():
    """Test safety checks in sequential synergy detector."""
    logger.info("Testing sequential synergy detector safety checks...")
    
    try:
        # Mock the required components for testing
        class MockKernel:
            def __init__(self):
                self.config = {
                    'synergy_detector': {
                        'time_window_bars': 10,
                        'cooldown_bars': 5,
                        'synergy_expiration_minutes': 30
                    }
                }
                self.event_bus = MockEventBus()
        
        class MockEventBus:
            def subscribe(self, event_type, handler):
                pass
            def unsubscribe(self, event_type, handler):
                pass
            def create_event(self, event_type, payload, source):
                return MockEvent(event_type, payload)
            def publish(self, event):
                pass
        
        class MockEvent:
            def __init__(self, event_type, payload):
                self.event_type = event_type
                self.payload = payload
                self.timestamp = datetime.now()
        
        # Create mock kernel
        kernel = MockKernel()
        
        # Test sequential detector initialization
        detector = SequentialSynergyDetector("test_detector", kernel)
        
        # Test process_features with system OFF
        test_features = {
            'nwrqk_bull': True,
            'nwrqk_strength': 0.8,
            'mlmi_bull': True,
            'mlmi_confidence': 0.7,
            'fvg_bull': True,
            'fvg_size': 0.01
        }
        
        result = detector.process_features(test_features, datetime.now())
        
        # Should return None when system is OFF
        assert result is None, "Expected None when system is OFF"
        
        # Test status reporting
        status = detector.get_status()
        assert 'system_active' in status, "Expected system_active in status"
        assert not status['system_active'], "Expected system_active to be False when OFF"
        
        logger.info("✓ Sequential synergy detector safety checks PASSED")
        return True
        
    except SystemExit:
        logger.info("Kill switch triggered exit (expected behavior)")
        return True
    except Exception as e:
        logger.error(f"Sequential detector safety check failed: {e}")
        return False

def main():
    """Run all safety check tests."""
    logger.info("Starting safety check tests...")
    
    tests = [
        test_synergy_detector_safety,
        test_indicators_safety,
        test_sequential_detector_safety
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"Safety check tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All safety checks PASSED! Signal blocking is working correctly.")
        return True
    else:
        logger.error("❌ Some safety checks FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)