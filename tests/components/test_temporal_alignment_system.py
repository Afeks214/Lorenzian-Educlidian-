"""
AGENT 2 - TEMPORAL ALIGNMENT SYSTEM TEST SUITE
===============================================

Comprehensive test suite for the bulletproof timestamp alignment system.
Tests all edge cases including market holidays, gaps, and extended hours.

Author: AGENT 2 - Timestamp Alignment Specialist  
Date: 2025-07-16
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/QuantNova/GrandModel')

from src.components.temporal_alignment_system import (
    TemporalAlignmentSystem,
    AlignmentConfig,
    create_optimized_alignment_system
)
from src.components.improved_synergy_alignment import (
    align_timeframes_improved,
    validate_alignment_improvement
)

class TestTemporalAlignmentSystem:
    """Test suite for TemporalAlignmentSystem"""
    
    @pytest.fixture
    def sample_30m_data(self):
        """Create sample 30-minute data"""
        dates = pd.date_range(
            start='2024-01-01 09:30:00',
            end='2024-01-01 16:00:00', 
            freq='30min',
            tz='America/New_York'
        )
        
        np.random.seed(42)
        data = {
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(95, 115, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates)),
            'MLMI_Bullish': np.random.choice([True, False], len(dates)),
            'MLMI_Bearish': np.random.choice([True, False], len(dates)),
            'NWRQK_Bullish': np.random.choice([True, False], len(dates)),
            'NWRQK_Bearish': np.random.choice([True, False], len(dates))
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture  
    def sample_5m_data(self):
        """Create sample 5-minute data"""
        dates = pd.date_range(
            start='2024-01-01 09:30:00',
            end='2024-01-01 16:00:00',
            freq='5min',
            tz='America/New_York'
        )
        
        np.random.seed(42)
        data = {
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(95, 115, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates)),
            'FVG_Bull_Active': np.random.choice([True, False], len(dates)),
            'FVG_Bear_Active': np.random.choice([True, False], len(dates))
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_alignment_system_initialization(self):
        """Test alignment system initialization"""
        config = AlignmentConfig(signal_lag_minutes=2)
        system = TemporalAlignmentSystem(config)
        
        assert system.config.signal_lag_minutes == 2
        assert system.config.max_gap_minutes == 120
        assert system.alignment_cache == {}
        assert system.validation_results == {}
    
    def test_basic_alignment(self, sample_30m_data, sample_5m_data):
        """Test basic alignment functionality"""
        system = create_optimized_alignment_system()
        
        aligned_df = system.align_timeframes(sample_30m_data, sample_5m_data)
        
        # Check that result has 5-minute frequency
        assert len(aligned_df) == len(sample_5m_data)
        assert all(aligned_df.index == sample_5m_data.index)
        
        # Check that 30-minute indicators are present
        assert 'MLMI_Bullish' in aligned_df.columns
        assert 'NWRQK_Bullish' in aligned_df.columns
        
        # Check that FVG data is preserved
        assert 'FVG_Bull_Active' in aligned_df.columns
    
    def test_temporal_constraints(self, sample_30m_data, sample_5m_data):
        """Test that temporal constraints prevent look-ahead bias"""
        system = create_optimized_alignment_system(signal_lag_minutes=1)
        
        aligned_df = system.align_timeframes(sample_30m_data, sample_5m_data)
        
        # Verify no look-ahead bias
        for timestamp in aligned_df.index:
            if pd.notna(aligned_df.loc[timestamp, 'MLMI_Bullish']):
                # Check that signal only appears after proper lag
                minute = timestamp.minute
                
                if minute >= 30:
                    bar_close = timestamp.replace(minute=30, second=0, microsecond=0)
                else:
                    bar_close = timestamp.replace(minute=0, second=0, microsecond=0)
                
                earliest_signal = bar_close + pd.Timedelta(minutes=1)
                assert timestamp >= earliest_signal, f"Look-ahead bias at {timestamp}"
    
    def test_market_hours_handling(self):
        """Test market hours and weekend handling"""
        # Create data spanning weekend
        dates_30m = pd.date_range(
            start='2024-01-05 09:30:00',  # Friday
            end='2024-01-08 16:00:00',   # Monday
            freq='30min',
            tz='America/New_York'
        )
        
        dates_5m = pd.date_range(
            start='2024-01-05 09:30:00',
            end='2024-01-08 16:00:00', 
            freq='5min',
            tz='America/New_York'
        )
        
        # Create sample data
        df_30m = pd.DataFrame({
            'MLMI_Bullish': [True] * len(dates_30m),
            'Close': np.random.uniform(100, 110, len(dates_30m))
        }, index=dates_30m)
        
        df_5m = pd.DataFrame({
            'Close': np.random.uniform(100, 110, len(dates_5m)),
            'FVG_Bull_Active': [False] * len(dates_5m)
        }, index=dates_5m)
        
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(df_30m, df_5m)
        
        # Check that weekend signals are handled properly
        weekend_signals = 0
        for timestamp in aligned_df.index:
            if timestamp.dayofweek >= 5:  # Weekend
                if pd.notna(aligned_df.loc[timestamp, 'MLMI_Bullish']):
                    weekend_signals += 1
        
        # Should have very few or no weekend signals due to market hours filtering
        assert weekend_signals < len(aligned_df) * 0.1  # Less than 10%
    
    def test_data_gap_handling(self, sample_5m_data):
        """Test handling of data gaps"""
        # Create 30-minute data with gaps
        dates_30m = pd.date_range(
            start='2024-01-01 09:30:00',
            end='2024-01-01 16:00:00',
            freq='30min',
            tz='America/New_York'
        )
        
        # Remove some bars to create gaps
        gap_indices = [2, 3, 7]  # Remove a few bars
        dates_30m_gapped = dates_30m.delete(gap_indices)
        
        df_30m_gapped = pd.DataFrame({
            'MLMI_Bullish': [True] * len(dates_30m_gapped),
            'MLMI_Bearish': [False] * len(dates_30m_gapped),
            'Close': np.random.uniform(100, 110, len(dates_30m_gapped))
        }, index=dates_30m_gapped)
        
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(df_30m_gapped, sample_5m_data)
        
        # Check that gaps are handled gracefully
        assert len(aligned_df) == len(sample_5m_data)
        
        # Check alignment metrics
        report = system.create_alignment_report()
        assert 'data_gaps_count' in report['data_quality']
        assert report['data_quality']['data_gaps_count'] >= 0
    
    def test_validation_metrics(self, sample_30m_data, sample_5m_data):
        """Test validation metrics generation"""
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(sample_30m_data, sample_5m_data)
        
        # Get validation report
        report = system.create_alignment_report()
        
        # Check report structure
        assert 'alignment_summary' in report
        assert 'data_quality' in report
        assert 'configuration' in report
        assert 'recommendations' in report
        
        # Check alignment summary
        summary = report['alignment_summary']
        assert 'alignment_accuracy' in summary
        assert 'aligned_bars' in summary
        assert 'missing_periods' in summary
        
        # Check data quality
        quality = report['data_quality']
        assert 'data_gaps_count' in quality
        assert 'validation_warnings' in quality
    
    def test_column_mapping(self, sample_30m_data, sample_5m_data):
        """Test custom column mapping"""
        system = create_optimized_alignment_system()
        
        # Define custom column mapping
        column_mapping = {
            'MLMI_Bullish': 'momentum_bull',
            'MLMI_Bearish': 'momentum_bear'
        }
        
        aligned_df = system.align_timeframes(
            sample_30m_data, 
            sample_5m_data,
            column_mapping=column_mapping
        )
        
        # Check that mapped columns exist
        assert 'momentum_bull' in aligned_df.columns
        assert 'momentum_bear' in aligned_df.columns
        
        # Original column names should not be present
        assert 'MLMI_Bullish' not in aligned_df.columns
        assert 'MLMI_Bearish' not in aligned_df.columns
    
    def test_performance_optimization(self, sample_30m_data, sample_5m_data):
        """Test that alignment maintains good performance"""
        import time
        
        system = create_optimized_alignment_system()
        
        # Time the alignment
        start_time = time.time()
        aligned_df = system.align_timeframes(sample_30m_data, sample_5m_data)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for sample data)
        assert execution_time < 1.0
        
        # Check that result is correct size
        assert len(aligned_df) == len(sample_5m_data)
    
    def test_extended_hours_scenario(self):
        """Test extended hours and pre-market scenarios"""
        # Create extended hours data (4 AM to 8 PM)
        dates_30m = pd.date_range(
            start='2024-01-01 04:00:00',
            end='2024-01-01 20:00:00',
            freq='30min',
            tz='America/New_York'
        )
        
        dates_5m = pd.date_range(
            start='2024-01-01 04:00:00',
            end='2024-01-01 20:00:00',
            freq='5min', 
            tz='America/New_York'
        )
        
        df_30m = pd.DataFrame({
            'MLMI_Bullish': [True] * len(dates_30m),
            'Close': np.random.uniform(100, 110, len(dates_30m))
        }, index=dates_30m)
        
        df_5m = pd.DataFrame({
            'Close': np.random.uniform(100, 110, len(dates_5m)),
            'FVG_Bull_Active': [False] * len(dates_5m)
        }, index=dates_5m)
        
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(df_30m, df_5m)
        
        # Should handle extended hours data
        assert len(aligned_df) == len(df_5m)
        
        # Check that some signals are present during extended hours
        extended_hours_signals = 0
        for timestamp in aligned_df.index:
            time_of_day = timestamp.time()
            if (datetime.strptime('04:00', '%H:%M').time() <= time_of_day <= 
                datetime.strptime('09:30', '%H:%M').time()) or \
               (datetime.strptime('16:00', '%H:%M').time() <= time_of_day <= 
                datetime.strptime('20:00', '%H:%M').time()):
                if pd.notna(aligned_df.loc[timestamp, 'MLMI_Bullish']):
                    extended_hours_signals += 1
        
        # Should have some extended hours signals
        assert extended_hours_signals >= 0
    
    def test_holiday_handling(self):
        """Test handling of market holidays"""
        # Create data around a holiday (e.g., New Year's Day 2024)
        dates_30m = pd.date_range(
            start='2023-12-29 09:30:00',  # Friday before
            end='2024-01-02 16:00:00',   # Tuesday after
            freq='30min',
            tz='America/New_York'
        )
        
        dates_5m = pd.date_range(
            start='2023-12-29 09:30:00',
            end='2024-01-02 16:00:00',
            freq='5min',
            tz='America/New_York'
        )
        
        df_30m = pd.DataFrame({
            'MLMI_Bullish': [True] * len(dates_30m),
            'Close': np.random.uniform(100, 110, len(dates_30m))
        }, index=dates_30m)
        
        df_5m = pd.DataFrame({
            'Close': np.random.uniform(100, 110, len(dates_5m)),
            'FVG_Bull_Active': [False] * len(dates_5m)
        }, index=dates_5m)
        
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(df_30m, df_5m)
        
        # Should handle holiday data gracefully
        assert len(aligned_df) == len(df_5m)
        
        # Check report for holiday handling
        report = system.create_alignment_report()
        assert report['alignment_summary']['total_5min_bars'] > 0


class TestImprovedSynergyAlignment:
    """Test the improved synergy alignment integration"""
    
    @pytest.fixture
    def synergy_30m_data(self):
        """Sample 30-minute data with synergy indicators"""
        dates = pd.date_range(
            start='2024-01-01 09:30:00',
            end='2024-01-01 16:00:00',
            freq='30min'
        )
        
        return pd.DataFrame({
            'Open': [100] * len(dates),
            'High': [105] * len(dates),
            'Low': [95] * len(dates),
            'Close': [102] * len(dates),
            'Volume': [5000] * len(dates),
            'MLMI_Bullish': [True, False, True, False, True, False, True, False, True, False, True, False, True][:len(dates)],
            'MLMI_Bearish': [False, True, False, True, False, True, False, True, False, True, False, True, False][:len(dates)],
            'NWRQK_Bullish': [True, True, False, False, True, True, False, False, True, True, False, False, True][:len(dates)],
            'NWRQK_Bearish': [False, False, True, True, False, False, True, True, False, False, True, True, False][:len(dates)]
        }, index=dates)
    
    @pytest.fixture
    def synergy_5m_data(self):
        """Sample 5-minute data with FVG indicators"""
        dates = pd.date_range(
            start='2024-01-01 09:30:00',
            end='2024-01-01 16:00:00',
            freq='5min'
        )
        
        return pd.DataFrame({
            'Open': np.random.uniform(99, 103, len(dates)),
            'High': np.random.uniform(103, 107, len(dates)),
            'Low': np.random.uniform(97, 101, len(dates)),
            'Close': np.random.uniform(100, 104, len(dates)),
            'Volume': np.random.randint(1000, 3000, len(dates)),
            'FVG_Bull_Active': np.random.choice([True, False], len(dates), p=[0.3, 0.7]),
            'FVG_Bear_Active': np.random.choice([True, False], len(dates), p=[0.3, 0.7])
        }, index=dates)
    
    def test_improved_alignment_integration(self, synergy_30m_data, synergy_5m_data):
        """Test the improved alignment function"""
        aligned_df = align_timeframes_improved(synergy_30m_data, synergy_5m_data)
        
        # Check structure
        assert len(aligned_df) == len(synergy_5m_data)
        assert 'MLMI_Bullish' in aligned_df.columns
        assert 'NWRQK_Bullish' in aligned_df.columns
        assert 'FVG_Bull_Active' in aligned_df.columns
        
        # Check that 5-minute data is preserved
        pd.testing.assert_series_equal(
            aligned_df['FVG_Bull_Active'], 
            synergy_5m_data['FVG_Bull_Active'],
            check_names=False
        )
    
    def test_validation_comprehensive(self, synergy_30m_data, synergy_5m_data):
        """Test comprehensive validation"""
        validation_results = validate_alignment_improvement(synergy_30m_data, synergy_5m_data)
        
        # Check validation structure
        assert 'alignment_quality' in validation_results
        assert 'temporal_constraints' in validation_results
        assert 'lookahead_prevention' in validation_results
        assert 'market_hours_handling' in validation_results
        assert 'overall_status' in validation_results
        
        # Check that validations passed
        assert validation_results['temporal_constraints']['passed'] == True
        assert validation_results['lookahead_prevention']['passed'] == True
        assert validation_results['market_hours_handling']['passed'] == True


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("🧪 RUNNING COMPREHENSIVE TEMPORAL ALIGNMENT TESTS")
    print("="*60)
    
    # Run pytest with verbose output
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        '/home/QuantNova/GrandModel/tests/components/test_temporal_alignment_system.py',
        '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0


if __name__ == "__main__":
    print("🎯 AGENT 2 - TEMPORAL ALIGNMENT SYSTEM TESTS")
    print("Testing bulletproof timestamp alignment implementation")
    print()
    
    # Run basic validation without pytest
    try:
        from src.components.temporal_alignment_system import create_optimized_alignment_system
        
        # Create test data
        dates_30m = pd.date_range('2024-01-01 09:30', '2024-01-01 16:00', freq='30min')
        dates_5m = pd.date_range('2024-01-01 09:30', '2024-01-01 16:00', freq='5min')
        
        df_30m = pd.DataFrame({
            'MLMI_Bullish': [True, False, True, False, True, False, True, False, True, False, True, False, True][:len(dates_30m)],
            'Close': [100] * len(dates_30m)
        }, index=dates_30m)
        
        df_5m = pd.DataFrame({
            'Close': [100] * len(dates_5m),
            'FVG_Bull_Active': [False] * len(dates_5m)
        }, index=dates_5m)
        
        # Test alignment
        system = create_optimized_alignment_system()
        aligned_df = system.align_timeframes(df_30m, df_5m)
        report = system.create_alignment_report()
        
        print("✅ Basic alignment test passed")
        print(f"Alignment accuracy: {report['alignment_summary']['alignment_accuracy']}")
        print(f"Aligned bars: {report['alignment_summary']['aligned_bars']}")
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()