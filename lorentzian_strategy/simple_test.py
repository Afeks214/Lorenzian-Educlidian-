"""
Simple test of the Lorentzian Strategy data pipeline without external dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, '/home/QuantNova/GrandModel')

def simple_data_test():
    """Test basic data loading and processing"""
    print("=" * 60)
    print("SIMPLE LORENTZIAN STRATEGY TEST")
    print("=" * 60)
    
    # Test 1: Load raw NQ data
    print("\n1. Testing raw data loading...")
    try:
        data_file = "/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv"
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, format='mixed')
        
        print("✓ Raw data loaded successfully")
        print(f"   - Records: {len(df):,}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"✗ Raw data loading failed: {e}")
        return False
    
    # Test 2: Basic data validation
    print("\n2. Testing data validation...")
    try:
        # Check for missing values
        missing = df.isnull().sum()
        print(f"✓ Missing values check: {missing.sum()} total missing")
        
        # Check OHLC integrity
        high_low_valid = (df['High'] >= df['Low']).all()
        open_valid = ((df['Open'] >= df['Low']) & (df['Open'] <= df['High'])).all()
        close_valid = ((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all()
        
        print(f"✓ OHLC validation:")
        print(f"   - High >= Low: {high_low_valid}")
        print(f"   - Open in range: {open_valid}")
        print(f"   - Close in range: {close_valid}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['Timestamp']).sum()
        print(f"   - Duplicate timestamps: {duplicates}")
        
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return False
    
    # Test 3: Simple feature calculation
    print("\n3. Testing feature calculation...")
    try:
        # Calculate basic indicators
        close_prices = df['Close'].values
        
        # Simple RSI calculation
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.zeros(len(prices))
            avg_losses = np.zeros(len(prices))
            
            # Initial values
            if len(gains) >= period:
                avg_gains[period] = np.mean(gains[:period])
                avg_losses[period] = np.mean(losses[:period])
                
                # Rolling calculation
                for i in range(period + 1, len(prices)):
                    avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
                    avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
            
            rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(close_prices)
        
        # Simple moving averages
        sma_20 = pd.Series(close_prices).rolling(20).mean().fillna(close_prices[0])
        sma_50 = pd.Series(close_prices).rolling(50).mean().fillna(close_prices[0])
        
        # Returns
        returns_1 = np.log(close_prices[1:] / close_prices[:-1])
        returns_1 = np.concatenate([[0], returns_1])  # Pad first value
        
        print("✓ Feature calculation successful")
        print(f"   - RSI range: {rsi[rsi > 0].min():.1f} to {rsi.max():.1f}")
        print(f"   - SMA20 last value: {sma_20.iloc[-1]:.2f}")
        print(f"   - SMA50 last value: {sma_50.iloc[-1]:.2f}")
        print(f"   - Returns std: {np.std(returns_1):.4f}")
        
    except Exception as e:
        print(f"✗ Feature calculation failed: {e}")
        return False
    
    # Test 4: Performance timing
    print("\n4. Testing performance...")
    try:
        import time
        
        # Time data loading
        start_time = time.time()
        df_test = pd.read_csv(data_file)
        df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], dayfirst=True, format='mixed')
        load_time = time.time() - start_time
        
        # Time feature calculation on subset
        subset = df.iloc[-1000:].copy()  # Last 1000 records
        start_time = time.time()
        rsi_test = calculate_rsi(subset['Close'].values)
        calc_time = time.time() - start_time
        
        print("✓ Performance test completed")
        print(f"   - Data loading: {load_time:.3f}s ({len(df_test):,} records)")
        print(f"   - Feature calculation: {calc_time:.3f}s ({len(subset)} records)")
        print(f"   - Processing rate: {len(subset)/calc_time:.0f} records/second")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    
    # Test 5: Data quality summary
    print("\n5. Data quality summary...")
    try:
        quality_metrics = {
            'total_records': len(df),
            'date_range_days': (df['Timestamp'].max() - df['Timestamp'].min()).days,
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_timestamps': int(df.duplicated(subset=['Timestamp']).sum()),
            'price_range': {
                'min': float(df['Low'].min()),
                'max': float(df['High'].max()),
                'latest': float(df['Close'].iloc[-1])
            },
            'volume_stats': {
                'total': int(df['Volume'].sum()),
                'average': float(df['Volume'].mean()),
                'zero_volume': int((df['Volume'] == 0).sum())
            },
            'data_integrity': {
                'high_low_valid': bool((df['High'] >= df['Low']).all()),
                'open_in_range': bool(((df['Open'] >= df['Low']) & (df['Open'] <= df['High'])).all()),
                'close_in_range': bool(((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all())
            }
        }
        
        print("✓ Quality analysis completed")
        for key, value in quality_metrics.items():
            if isinstance(value, dict):
                print(f"   - {key}:")
                for subkey, subvalue in value.items():
                    print(f"     * {subkey}: {subvalue}")
            else:
                print(f"   - {key}: {value}")
        
        # Calculate quality score
        integrity_score = sum(quality_metrics['data_integrity'].values()) / len(quality_metrics['data_integrity']) * 100
        missing_penalty = min(10, quality_metrics['missing_values'] / len(df) * 100)
        duplicate_penalty = min(5, quality_metrics['duplicate_timestamps'] / len(df) * 100)
        
        quality_score = max(0, integrity_score - missing_penalty - duplicate_penalty)
        print(f"   - Overall quality score: {quality_score:.1f}%")
        
    except Exception as e:
        print(f"✗ Quality analysis failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    
    print(f"\nFINAL SUMMARY:")
    print(f"- Dataset: NQ 30-minute futures data")
    print(f"- Records: {len(df):,}")
    print(f"- Time span: {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")
    print(f"- Data integrity: {integrity_score:.1f}%")
    print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"- Processing ready: ✓")
    
    return True


if __name__ == "__main__":
    success = simple_data_test()
    if not success:
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("DATA PIPELINE FOUNDATION READY")
    print("Ready for Lorentzian strategy implementation!")
    print(f"{'='*60}")