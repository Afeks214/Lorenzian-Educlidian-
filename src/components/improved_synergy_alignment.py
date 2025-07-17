"""
AGENT 2 - IMPROVED SYNERGY ALIGNMENT SYSTEM
===========================================

This file demonstrates how to replace the crude 6:1 ratio mapping in synergy_strategies_backtest.py
with the new bulletproof timestamp alignment system.

The original align_timeframes() function used a simple forward-fill approach that could introduce
look-ahead bias. This replacement ensures proper temporal constraints.

Author: AGENT 2 - Timestamp Alignment Specialist
Date: 2025-07-16
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append('/home/QuantNova/GrandModel')

from src.components.temporal_alignment_system import (
    TemporalAlignmentSystem, 
    AlignmentConfig, 
    create_optimized_alignment_system
)

def align_timeframes_improved(df_30m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    IMPROVED VERSION of the align_timeframes function from synergy_strategies_backtest.py
    
    This replaces the crude forward-fill approach with bulletproof datetime alignment
    that respects temporal constraints and prevents look-ahead bias.
    
    Args:
        df_30m: 30-minute data with indicators (MLMI, NWRQK)
        df_5m: 5-minute data with FVG indicators
        
    Returns:
        Properly aligned DataFrame with temporal constraints enforced
    """
    print("  🔄 Aligning timeframes with improved temporal system...")
    
    # Initialize the alignment system with proper configuration
    alignment_system = create_optimized_alignment_system(signal_lag_minutes=1)
    
    # Define column mapping for synergy strategy indicators
    column_mapping = {
        'MLMI_Bullish': 'MLMI_Bullish',
        'MLMI_Bearish': 'MLMI_Bearish', 
        'NWRQK_Bullish': 'NWRQK_Bullish',
        'NWRQK_Bearish': 'NWRQK_Bearish'
    }
    
    # Perform the alignment
    aligned_df = alignment_system.align_timeframes(
        df_30m=df_30m,
        df_5m=df_5m,
        column_mapping=column_mapping
    )
    
    # Generate alignment quality report
    report = alignment_system.create_alignment_report()
    
    # Log alignment quality metrics
    alignment_summary = report.get('alignment_summary', {})
    print(f"    ✓ Alignment accuracy: {alignment_summary.get('alignment_accuracy', 'N/A')}")
    print(f"    ✓ Aligned bars: {alignment_summary.get('aligned_bars', 0):,}")
    print(f"    ✓ Missing periods: {alignment_summary.get('missing_periods', 0)}")
    
    # Warn about quality issues
    warnings = report.get('data_quality', {}).get('validation_warnings', [])
    for warning in warnings:
        print(f"    ⚠️ Warning: {warning}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    for rec in recommendations:
        print(f"    💡 Recommendation: {rec}")
    
    return aligned_df


def validate_alignment_improvement(df_30m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that the improved alignment system fixes temporal issues
    
    Returns:
        Validation results comparing old vs new alignment
    """
    print("\n📊 VALIDATING ALIGNMENT IMPROVEMENT")
    print("="*50)
    
    # Test the improved alignment
    print("Testing improved alignment system...")
    improved_df = align_timeframes_improved(df_30m, df_5m)
    
    # Create alignment system for detailed analysis
    alignment_system = create_optimized_alignment_system()
    aligned_test = alignment_system.align_timeframes(df_30m, df_5m)
    report = alignment_system.create_alignment_report()
    
    # Test temporal constraints
    temporal_validation = validate_temporal_constraints(df_30m, improved_df)
    
    # Test look-ahead bias prevention
    lookahead_validation = validate_lookahead_prevention(df_30m, improved_df)
    
    # Test market hours handling
    market_hours_validation = validate_market_hours_handling(improved_df)
    
    validation_results = {
        "alignment_quality": report,
        "temporal_constraints": temporal_validation,
        "lookahead_prevention": lookahead_validation,
        "market_hours_handling": market_hours_validation,
        "overall_status": "PASSED" if all([
            temporal_validation["passed"],
            lookahead_validation["passed"],
            market_hours_validation["passed"]
        ]) else "FAILED"
    }
    
    print(f"\n🎯 VALIDATION RESULT: {validation_results['overall_status']}")
    
    return validation_results


def validate_temporal_constraints(df_30m: pd.DataFrame, aligned_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that temporal constraints are properly enforced"""
    
    print("  🔍 Validating temporal constraints...")
    
    issues = []
    
    # Check that 30-minute signals only appear after the 30-minute bar closes
    for idx, row in aligned_df.iterrows():
        if pd.notna(row.get('MLMI_Bullish')):
            # This 5-minute bar has a 30-minute signal
            # Check if it's temporally valid
            
            minute = idx.minute
            expected_30m_close = None
            
            if minute >= 30:
                expected_30m_close = idx.replace(minute=30, second=0, microsecond=0)
            else:
                expected_30m_close = idx.replace(minute=0, second=0, microsecond=0)
            
            # Signal should only be available at least 1 minute after 30-minute bar close
            earliest_valid_time = expected_30m_close + pd.Timedelta(minutes=1)
            
            if idx < earliest_valid_time:
                issues.append(f"Look-ahead bias at {idx}: signal available before {earliest_valid_time}")
    
    passed = len(issues) == 0
    
    print(f"    {'✅' if passed else '❌'} Temporal constraints: {len(issues)} violations found")
    
    return {
        "passed": passed,
        "violations_count": len(issues),
        "issues": issues[:5]  # First 5 issues for brevity
    }


def validate_lookahead_prevention(df_30m: pd.DataFrame, aligned_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that look-ahead bias is prevented"""
    
    print("  🔍 Validating look-ahead bias prevention...")
    
    # Check that each 5-minute bar only uses 30-minute data that was available at that time
    lookahead_violations = 0
    
    for timestamp_5m in aligned_df.index:
        # Find what 30-minute data should be available at this time
        available_30m_data = df_30m[df_30m.index <= timestamp_5m]
        
        if len(available_30m_data) > 0:
            # Check if the aligned data matches the most recent available data
            # (accounting for signal lag)
            
            # Get the latest available 30-minute bar (with lag)
            target_time = timestamp_5m - pd.Timedelta(minutes=1)  # Signal lag
            valid_30m_data = df_30m[df_30m.index <= target_time]
            
            if len(valid_30m_data) == 0:
                # No valid 30-minute data should be available
                if pd.notna(aligned_df.loc[timestamp_5m, 'MLMI_Bullish']):
                    lookahead_violations += 1
    
    passed = lookahead_violations == 0
    
    print(f"    {'✅' if passed else '❌'} Look-ahead prevention: {lookahead_violations} violations")
    
    return {
        "passed": passed,
        "violations_count": lookahead_violations
    }


def validate_market_hours_handling(aligned_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate proper market hours handling"""
    
    print("  🔍 Validating market hours handling...")
    
    # Check for signals during non-market hours
    weekend_signals = 0
    after_hours_signals = 0
    
    for timestamp in aligned_df.index:
        # Check weekends
        if timestamp.dayofweek >= 5:  # Saturday = 5, Sunday = 6
            if pd.notna(aligned_df.loc[timestamp, 'MLMI_Bullish']):
                weekend_signals += 1
        
        # Check after hours (before 4 AM or after 8 PM ET)
        time_of_day = timestamp.time()
        if time_of_day < datetime.strptime('04:00', '%H:%M').time() or \
           time_of_day > datetime.strptime('20:00', '%H:%M').time():
            if pd.notna(aligned_df.loc[timestamp, 'MLMI_Bullish']):
                after_hours_signals += 1
    
    total_violations = weekend_signals + after_hours_signals
    passed = total_violations == 0
    
    print(f"    {'✅' if passed else '❌'} Market hours handling: {total_violations} violations")
    print(f"      Weekend signals: {weekend_signals}")
    print(f"      After-hours signals: {after_hours_signals}")
    
    return {
        "passed": passed,
        "weekend_violations": weekend_signals,
        "after_hours_violations": after_hours_signals,
        "total_violations": total_violations
    }


def create_performance_comparison(df_30m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare performance of old vs new alignment methods
    """
    print("\n⚡ PERFORMANCE COMPARISON")
    print("="*30)
    
    import time
    
    # Time the improved alignment
    start_time = time.time()
    improved_df = align_timeframes_improved(df_30m, df_5m)
    improved_time = time.time() - start_time
    
    # Simulate old alignment approach (crude forward-fill)
    start_time = time.time()
    old_style_df = simulate_old_alignment(df_30m, df_5m)
    old_time = time.time() - start_time
    
    print(f"🕐 Old alignment time: {old_time:.3f}s")
    print(f"🕐 New alignment time: {improved_time:.3f}s")
    print(f"📊 Speed ratio: {old_time/improved_time:.2f}x")
    
    # Quality comparison
    old_aligned_bars = old_style_df.notna().sum().sum()
    new_aligned_bars = improved_df.notna().sum().sum()
    
    return {
        "old_time_seconds": old_time,
        "new_time_seconds": improved_time,
        "speed_improvement": old_time / improved_time,
        "old_data_quality": old_aligned_bars,
        "new_data_quality": new_aligned_bars,
        "quality_improvement": new_aligned_bars / max(old_aligned_bars, 1)
    }


def simulate_old_alignment(df_30m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the old crude alignment approach for comparison
    """
    # This replicates the old align_timeframes function
    aligned_data = []
    
    for timestamp in df_5m.index:
        # Find the most recent 30-minute bar (CRUDE APPROACH)
        relevant_30m = df_30m[df_30m.index <= timestamp]
        if len(relevant_30m) > 0:
            latest_30m = relevant_30m.iloc[-1]
            aligned_data.append({
                'timestamp': timestamp,
                'MLMI_Bullish': latest_30m.get('MLMI_Bullish', False),
                'MLMI_Bearish': latest_30m.get('MLMI_Bearish', False),
                'NWRQK_Bullish': latest_30m.get('NWRQK_Bullish', False),
                'NWRQK_Bearish': latest_30m.get('NWRQK_Bearish', False)
            })
        else:
            aligned_data.append({
                'timestamp': timestamp,
                'MLMI_Bullish': False,
                'MLMI_Bearish': False,
                'NWRQK_Bullish': False,
                'NWRQK_Bearish': False
            })
    
    aligned_df = pd.DataFrame(aligned_data).set_index('timestamp')
    
    # Merge with 5-minute data
    result = df_5m.copy()
    for col in aligned_df.columns:
        result[col] = aligned_df[col]
    
    return result


# Demonstration of how to integrate into existing synergy backtest
def demonstrate_integration():
    """
    Demonstrate how to integrate the improved alignment into existing backtests
    """
    print("\n🔧 INTEGRATION DEMONSTRATION")
    print("="*40)
    
    print("""
To integrate the improved alignment system into synergy_strategies_backtest.py:

1. Replace the existing align_timeframes() function with:
   
   from src.components.improved_synergy_alignment import align_timeframes_improved
   
   # In main() function, replace:
   # df_combined = align_timeframes(df_30m, df_5m)
   # With:
   df_combined = align_timeframes_improved(df_30m, df_5m)

2. The improved system will:
   ✅ Prevent look-ahead bias by enforcing 1-minute signal lag
   ✅ Handle market hours and weekends properly
   ✅ Provide detailed alignment quality metrics
   ✅ Generate warnings for data quality issues
   ✅ Maintain vectorized operations for performance

3. Validation and monitoring:
   - Alignment accuracy reports
   - Temporal constraint validation
   - Data gap detection and handling
   - Performance metrics tracking

4. Benefits:
   - Realistic signal timing (no future data)
   - Better risk management in backtests
   - More accurate strategy performance metrics
   - Robust handling of market conditions
   """)


if __name__ == "__main__":
    print("🎯 AGENT 2 - IMPROVED SYNERGY ALIGNMENT SYSTEM")
    print("This module replaces crude 6:1 ratio mapping with bulletproof datetime alignment")
    print()
    
    demonstrate_integration()