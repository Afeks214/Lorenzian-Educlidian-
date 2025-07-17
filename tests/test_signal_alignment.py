"""
Comprehensive Test Suite for Signal Alignment System

This test suite validates the signal alignment system and demonstrates
the fixes for timeframe misalignment issues.

Tests include:
1. Signal interpolation accuracy
2. Timeframe synchronization
3. Signal validation and confidence scoring
4. Deterministic signal ordering
5. Look-ahead bias detection
6. Complete strategy integration

Author: Claude (Anthropic)
Date: 2025-01-17
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.signal_alignment import (
    SignalAlignmentEngine, SignalType, SignalDirection,
    TimeframeConverter, SignalStandardizer, SignalValidator,
    SignalData, create_signal_alignment_engine
)
from src.strategy.unified_signal_strategy import (
    UnifiedSignalStrategy, create_unified_strategy, create_default_strategy_config
)
from src.core.minimal_dependencies import EventBus, BarData


class TestSignalAlignment(unittest.TestCase):
    """Test cases for signal alignment system"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = create_signal_alignment_engine()
        self.converter = TimeframeConverter()
        self.standardizer = SignalStandardizer()
        self.validator = SignalValidator()
        
        # Create test data
        self.test_timestamps_30m = [
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 11, 0)
        ]
        
        self.test_timestamps_5m = [
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 9, 5),
            datetime(2024, 1, 1, 9, 10),
            datetime(2024, 1, 1, 9, 15),
            datetime(2024, 1, 1, 9, 20),
            datetime(2024, 1, 1, 9, 25),
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 1, 1, 9, 35),
            datetime(2024, 1, 1, 9, 40),
            datetime(2024, 1, 1, 9, 45),
            datetime(2024, 1, 1, 9, 50),
            datetime(2024, 1, 1, 9, 55),
            datetime(2024, 1, 1, 10, 0)
        ]
    
    def test_signal_standardization(self):
        """Test signal standardization across different types"""
        print("\n=== Testing Signal Standardization ===")
        
        # Test MLMI signal standardization
        mlmi_signal = self.standardizer.standardize_signal(
            SignalType.MLMI, 2.5, "30m", {"test": "data"}
        )
        
        self.assertEqual(mlmi_signal.signal_type, SignalType.MLMI)
        self.assertEqual(mlmi_signal.direction, SignalDirection.BULLISH)
        self.assertGreater(mlmi_signal.strength, 0.6)  # Should be strong signal
        self.assertGreater(mlmi_signal.confidence, 0.5)
        
        print(f"✓ MLMI signal standardized: strength={mlmi_signal.strength:.3f}, confidence={mlmi_signal.confidence:.3f}")
        
        # Test NW-RQK signal standardization
        nwrqk_signal = self.standardizer.standardize_signal(
            SignalType.NWRQK, -0.025, "30m"
        )
        
        self.assertEqual(nwrqk_signal.direction, SignalDirection.BEARISH)
        self.assertGreater(nwrqk_signal.strength, 0.4)
        
        print(f"✓ NW-RQK signal standardized: strength={nwrqk_signal.strength:.3f}, confidence={nwrqk_signal.confidence:.3f}")
        
        # Test FVG signal standardization
        fvg_signal = self.standardizer.standardize_signal(
            SignalType.FVG, 0.15, "5m"
        )
        
        self.assertEqual(fvg_signal.direction, SignalDirection.BULLISH)
        self.assertEqual(fvg_signal.timeframe, "5m")
        
        print(f"✓ FVG signal standardized: strength={fvg_signal.strength:.3f}, confidence={fvg_signal.confidence:.3f}")
    
    def test_timeframe_conversion(self):
        """Test 30m to 5m timeframe conversion"""
        print("\n=== Testing Timeframe Conversion ===")
        
        # Create 30m signals
        signals_30m = []
        for i, ts in enumerate(self.test_timestamps_30m):
            signal = SignalData(
                signal_type=SignalType.MLMI,
                direction=SignalDirection.BULLISH if i % 2 == 0 else SignalDirection.BEARISH,
                strength=0.8,
                confidence=0.7,
                timestamp=ts,
                timeframe="30m",
                raw_value=1.0 if i % 2 == 0 else -1.0,
                threshold=0.5,
                metadata={"test": f"signal_{i}"}
            )
            signals_30m.append(signal)
        
        # Convert to 5m
        signals_5m = self.converter.convert_30m_to_5m(signals_30m, self.test_timestamps_5m)
        
        # Validate results
        self.assertEqual(len(signals_5m), len(self.test_timestamps_5m))
        
        # Check that signals are properly interpolated
        for signal in signals_5m:
            self.assertEqual(signal.timeframe, "5m")
            self.assertLessEqual(signal.confidence, 0.7)  # Should decay with age
            self.assertTrue(signal.metadata.get('interpolated', False))
        
        print(f"✓ Converted {len(signals_30m)} 30m signals to {len(signals_5m)} 5m signals")
        
        # Test confidence decay
        first_signal = signals_5m[0]
        last_signal = signals_5m[-1]
        
        # First signal should have higher confidence (no age)
        self.assertGreaterEqual(first_signal.confidence, last_signal.confidence)
        
        print(f"✓ Confidence decay working: first={first_signal.confidence:.3f}, last={last_signal.confidence:.3f}")
    
    def test_signal_validation(self):
        """Test signal validation logic"""
        print("\n=== Testing Signal Validation ===")
        
        # Test valid signal
        valid_signal = SignalData(
            signal_type=SignalType.MLMI,
            direction=SignalDirection.BULLISH,
            strength=0.8,
            confidence=0.7,
            timestamp=datetime.now(),
            timeframe="30m",
            raw_value=1.5,
            threshold=0.5,
            metadata={}
        )
        
        is_valid, error = self.validator.validate_signal(valid_signal)
        self.assertTrue(is_valid)
        print(f"✓ Valid signal passed validation: {error}")
        
        # Test invalid signal (strength out of range)
        invalid_signal = SignalData(
            signal_type=SignalType.MLMI,
            direction=SignalDirection.BULLISH,
            strength=1.5,  # Invalid: > 1.0
            confidence=0.7,
            timestamp=datetime.now(),
            timeframe="30m",
            raw_value=1.5,
            threshold=0.5,
            metadata={}
        )
        
        is_valid, error = self.validator.validate_signal(invalid_signal)
        self.assertFalse(is_valid)
        self.assertIn("strength", error.lower())
        print(f"✓ Invalid signal rejected: {error}")
        
        # Test look-ahead bias detection
        signals_with_bias = [
            SignalData(
                signal_type=SignalType.FVG,
                direction=SignalDirection.BULLISH,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime(2024, 1, 1, 9, 0),
                timeframe="5m",
                raw_value=1.0,
                threshold=0.5,
                metadata={}
            ),
            SignalData(
                signal_type=SignalType.FVG,
                direction=SignalDirection.BULLISH,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime(2024, 1, 1, 9, 20),  # 20 minutes later (suspicious)
                timeframe="5m",
                raw_value=1.0,
                threshold=0.5,
                metadata={}
            )
        ]
        
        clean_signals = self.validator.detect_look_ahead_bias(signals_with_bias)
        self.assertEqual(len(clean_signals), 1)  # Should remove the biased signal
        print(f"✓ Look-ahead bias detection working: {len(signals_with_bias)} → {len(clean_signals)} signals")
    
    def test_signal_alignment_engine(self):
        """Test complete signal alignment engine"""
        print("\n=== Testing Signal Alignment Engine ===")
        
        # Process signals of different types
        mlmi_signal = self.engine.process_raw_signal(
            SignalType.MLMI, 2.0, "30m", datetime.now(), {"test": "mlmi"}
        )
        self.assertIsNotNone(mlmi_signal)
        
        nwrqk_signal = self.engine.process_raw_signal(
            SignalType.NWRQK, -0.03, "30m", datetime.now(), {"test": "nwrqk"}
        )
        self.assertIsNotNone(nwrqk_signal)
        
        fvg_signal = self.engine.process_raw_signal(
            SignalType.FVG, 0.25, "5m", datetime.now(), {"test": "fvg"}
        )
        self.assertIsNotNone(fvg_signal)
        
        # Test synchronized signal retrieval
        current_time = datetime.now()
        synchronized = self.engine.get_synchronized_signals(current_time)
        
        self.assertIn(SignalType.MLMI, synchronized)
        self.assertIn(SignalType.NWRQK, synchronized)
        self.assertIn(SignalType.FVG, synchronized)
        
        print(f"✓ Processed 3 signals, synchronized {len(synchronized)} signals")
        
        # Test statistics
        stats = self.engine.get_stats()
        self.assertGreater(stats['signals_processed'], 0)
        
        print(f"✓ Engine stats: {stats}")
    
    def test_strategy_integration(self):
        """Test complete strategy integration"""
        print("\n=== Testing Strategy Integration ===")
        
        # Create strategy
        config = create_default_strategy_config()
        event_bus = EventBus()
        strategy = create_unified_strategy(config, event_bus)
        
        # Create test bar data
        test_bars_5m = []
        test_bars_30m = []
        
        # Generate 5-minute bars
        for i, ts in enumerate(self.test_timestamps_5m):
            bar = BarData(
                timestamp=ts,
                open=100.0 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.0 + i * 0.1,
                close=100.5 + i * 0.1,
                volume=1000 + i * 10
            )
            test_bars_5m.append(bar)
        
        # Generate 30-minute bars
        for i, ts in enumerate(self.test_timestamps_30m):
            bar = BarData(
                timestamp=ts,
                open=100.0 + i * 0.5,
                high=101.0 + i * 0.5,
                low=99.0 + i * 0.5,
                close=100.25 + i * 0.5,
                volume=5000 + i * 100
            )
            test_bars_30m.append(bar)
        
        # Process bars through strategy
        results_5m = []
        results_30m = []
        
        # Process 30m bars first to generate base signals
        for bar in test_bars_30m:
            result = strategy.process_30m_bar(bar)
            results_30m.append(result)
            print(f"✓ Processed 30m bar: {bar.timestamp}")
        
        # Process 5m bars
        for bar in test_bars_5m:
            result = strategy.process_5m_bar(bar)
            results_5m.append(result)
            print(f"✓ Processed 5m bar: {bar.timestamp}")
        
        # Validate results
        self.assertEqual(len(results_5m), len(test_bars_5m))
        self.assertEqual(len(results_30m), len(test_bars_30m))
        
        # Check that strategy statistics are updated
        final_stats = strategy._get_strategy_stats()
        self.assertGreater(final_stats['signals_processed'], 0)
        
        print(f"✓ Strategy integration successful: {final_stats}")
        
        # Test performance report
        performance = strategy.get_performance_report()
        if 'error' not in performance:
            print(f"✓ Performance report generated: {performance}")
        else:
            print(f"✓ No trades yet (expected for test data): {performance['error']}")
    
    def test_mapping_indices_fix(self):
        """Test that the new system fixes the mapping_indices calculation issues"""
        print("\n=== Testing Mapping Indices Fix ===")
        
        # Simulate the old problematic mapping
        old_mapping = np.array([min(i // 6, 10) for i in range(60)])  # 60 5m bars, 10 30m bars
        
        # Test edge cases that caused issues
        print(f"Old mapping sample: {old_mapping[:15]}")
        print(f"Old mapping issues: Last few values = {old_mapping[-5:]}")
        
        # Test new interpolation-based approach
        base_time = datetime(2024, 1, 1, 9, 0)
        timestamps_30m = [base_time + timedelta(minutes=30*i) for i in range(10)]
        timestamps_5m = [base_time + timedelta(minutes=5*i) for i in range(60)]
        
        # Create test signals
        signals_30m = []
        for i, ts in enumerate(timestamps_30m):
            signal = self.standardizer.standardize_signal(
                SignalType.MLMI, 1.0 + i * 0.1, "30m"
            )
            signal.timestamp = ts
            signals_30m.append(signal)
        
        # Convert using new system
        signals_5m = self.converter.convert_30m_to_5m(signals_30m, timestamps_5m)
        
        # Validate no index out of bounds
        self.assertEqual(len(signals_5m), len(timestamps_5m))
        
        # Check temporal consistency
        for i, signal in enumerate(signals_5m):
            self.assertEqual(signal.timestamp, timestamps_5m[i])
            self.assertLessEqual(signal.confidence, 1.0)
        
        print(f"✓ New interpolation system handles {len(timestamps_5m)} 5m timestamps correctly")
        print(f"✓ No index out of bounds errors")
        print(f"✓ Temporal consistency maintained")
    
    def test_signal_ordering_deterministic(self):
        """Test that signal ordering is deterministic"""
        print("\n=== Testing Deterministic Signal Ordering ===")
        
        # Create signals with same timestamp but different types
        base_time = datetime.now()
        signals = []
        
        # Add signals in random order
        signal_types = [SignalType.FVG, SignalType.MLMI, SignalType.NWRQK]
        
        for i, signal_type in enumerate(signal_types):
            signal = self.engine.process_raw_signal(
                signal_type, 1.0, "5m", base_time, {"order": i}
            )
            signals.append(signal)
        
        # Process multiple times to ensure deterministic ordering
        results1 = []
        results2 = []
        
        for _ in range(3):
            signal = self.engine.signal_queue.get_next_signal()
            if signal:
                results1.append(signal.signal_type)
        
        # Reset and do again
        self.engine.signal_queue.clear()
        
        for i, signal_type in enumerate(signal_types):
            signal = self.engine.process_raw_signal(
                signal_type, 1.0, "5m", base_time, {"order": i}
            )
        
        for _ in range(3):
            signal = self.engine.signal_queue.get_next_signal()
            if signal:
                results2.append(signal.signal_type)
        
        # Results should be deterministic
        self.assertEqual(results1, results2)
        print(f"✓ Signal ordering is deterministic: {[s.value for s in results1]}")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("COMPREHENSIVE SIGNAL ALIGNMENT TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test methods
    test_methods = [
        'test_signal_standardization',
        'test_timeframe_conversion',
        'test_signal_validation',
        'test_signal_alignment_engine',
        'test_strategy_integration',
        'test_mapping_indices_fix',
        'test_signal_ordering_deterministic'
    ]
    
    for method in test_methods:
        test_suite.addTest(TestSignalAlignment(method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - SIGNAL ALIGNMENT SYSTEM IS WORKING CORRECTLY!")
    else:
        print("\n❌ Some tests failed - please review the output above")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)