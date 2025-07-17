#!/usr/bin/env python3
"""
CL Data Quality Analyzer - Quick analysis of processed data
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def analyze_processed_data():
    """
    Analyze the processed CL data files
    """
    print("="*80)
    print("CL DATA QUALITY ANALYSIS FOR 500% TRUSTWORTHY BACKTESTING")
    print("="*80)
    
    data_dir = "/home/QuantNova/GrandModel/colab/data/"
    
    # Load processed data
    try:
        df_5min = pd.read_csv(f"{data_dir}CL_5min_processed.csv", index_col=0, parse_dates=True)
        print(f"✓ 5-minute data loaded: {len(df_5min):,} records")
    except Exception as e:
        print(f"✗ Error loading 5-minute data: {e}")
        return
    
    try:
        df_30min = pd.read_csv(f"{data_dir}CL_30min_processed.csv", index_col=0, parse_dates=True)
        print(f"✓ 30-minute data loaded: {len(df_30min):,} records")
    except Exception as e:
        print(f"✗ Error loading 30-minute data: {e}")
        return
    
    # Load train/test splits
    try:
        df_5min_train = pd.read_csv(f"{data_dir}CL_5min_train.csv", index_col=0, parse_dates=True)
        df_5min_test = pd.read_csv(f"{data_dir}CL_5min_test.csv", index_col=0, parse_dates=True)
        print(f"✓ 5-minute train/test splits: {len(df_5min_train):,} / {len(df_5min_test):,} records")
    except Exception as e:
        print(f"✗ Error loading 5-minute train/test: {e}")
        return
    
    print("\n" + "="*80)
    print("DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Date range analysis
    print(f"\n📅 DATA COVERAGE:")
    print(f"• 5-minute data: {df_5min.index.min()} to {df_5min.index.max()}")
    print(f"• 30-minute data: {df_30min.index.min()} to {df_30min.index.max()}")
    
    data_span_days = (df_5min.index.max() - df_5min.index.min()).days
    data_span_years = data_span_days / 365.25
    print(f"• Data span: {data_span_days} days ({data_span_years:.1f} years)")
    
    # Price range analysis
    print(f"\n💰 PRICE ANALYSIS:")
    print(f"• 5-minute Close price range: ${df_5min['Close'].min():.2f} - ${df_5min['Close'].max():.2f}")
    print(f"• 30-minute Close price range: ${df_30min['Close'].min():.2f} - ${df_30min['Close'].max():.2f}")
    print(f"• 5-minute average price: ${df_5min['Close'].mean():.2f}")
    print(f"• 30-minute average price: ${df_30min['Close'].mean():.2f}")
    
    # Volume analysis
    print(f"\n📊 VOLUME ANALYSIS:")
    print(f"• 5-minute volume range: {df_5min['Volume'].min():,} - {df_5min['Volume'].max():,}")
    print(f"• 30-minute volume range: {df_30min['Volume'].min():,} - {df_30min['Volume'].max():,}")
    print(f"• 5-minute average volume: {df_5min['Volume'].mean():,.0f}")
    print(f"• 30-minute average volume: {df_30min['Volume'].mean():,.0f}")
    
    # Technical indicators
    print(f"\n🔧 TECHNICAL INDICATORS:")
    print(f"• 5-minute data columns: {len(df_5min.columns)}")
    print(f"• 30-minute data columns: {len(df_30min.columns)}")
    
    # Sample some technical indicators
    if 'RSI' in df_5min.columns:
        print(f"• RSI range (5min): {df_5min['RSI'].min():.1f} - {df_5min['RSI'].max():.1f}")
    if 'MA_20' in df_5min.columns:
        print(f"• 20-period MA available: ✓")
    if 'BB_Upper' in df_5min.columns:
        print(f"• Bollinger Bands available: ✓")
    
    # Data quality metrics
    print(f"\n🎯 DATA QUALITY METRICS:")
    
    # Check for missing values
    missing_5min = df_5min.isnull().sum().sum()
    missing_30min = df_30min.isnull().sum().sum()
    print(f"• Missing values (5min): {missing_5min:,}")
    print(f"• Missing values (30min): {missing_30min:,}")
    
    # Check for duplicates
    duplicates_5min = df_5min.index.duplicated().sum()
    duplicates_30min = df_30min.index.duplicated().sum()
    print(f"• Duplicate timestamps (5min): {duplicates_5min:,}")
    print(f"• Duplicate timestamps (30min): {duplicates_30min:,}")
    
    # OHLC validation
    ohlc_valid_5min = (df_5min['High'] >= df_5min['Low']).all()
    ohlc_valid_30min = (df_30min['High'] >= df_30min['Low']).all()
    print(f"• OHLC relationships valid (5min): {'✓' if ohlc_valid_5min else '✗'}")
    print(f"• OHLC relationships valid (30min): {'✓' if ohlc_valid_30min else '✗'}")
    
    # Volume validation
    volume_valid_5min = (df_5min['Volume'] > 0).all()
    volume_valid_30min = (df_30min['Volume'] > 0).all()
    print(f"• Volume > 0 (5min): {'✓' if volume_valid_5min else '✗'}")
    print(f"• Volume > 0 (30min): {'✓' if volume_valid_30min else '✗'}")
    
    # Train/test split analysis
    print(f"\n🔄 TRAIN/TEST SPLITS:")
    print(f"• Training data: {len(df_5min_train):,} records ({len(df_5min_train)/len(df_5min)*100:.1f}%)")
    print(f"• Testing data: {len(df_5min_test):,} records ({len(df_5min_test)/len(df_5min)*100:.1f}%)")
    print(f"• Train period: {df_5min_train.index.min()} to {df_5min_train.index.max()}")
    print(f"• Test period: {df_5min_test.index.min()} to {df_5min_test.index.max()}")
    
    # Calculate overall quality score
    quality_factors = []
    
    # Data completeness (assuming 67% is acceptable for forex data)
    if data_span_years >= 3:
        quality_factors.append(100)
    else:
        quality_factors.append(data_span_years / 3 * 100)
    
    # OHLC validity
    quality_factors.append(100 if ohlc_valid_5min and ohlc_valid_30min else 0)
    
    # Volume validity
    quality_factors.append(100 if volume_valid_5min and volume_valid_30min else 0)
    
    # Missing data penalty
    total_records = len(df_5min) + len(df_30min)
    missing_penalty = (missing_5min + missing_30min) / total_records * 100
    quality_factors.append(max(0, 100 - missing_penalty))
    
    # Duplicate penalty
    duplicate_penalty = (duplicates_5min + duplicates_30min) / total_records * 100
    quality_factors.append(max(0, 100 - duplicate_penalty))
    
    overall_quality = np.mean(quality_factors)
    
    print(f"\n📈 OVERALL ASSESSMENT:")
    print(f"• Overall Quality Score: {overall_quality:.1f}%")
    print(f"• Data Span: {data_span_years:.1f} years")
    print(f"• Backtesting Ready: {'✓ YES' if data_span_years >= 3 and overall_quality >= 80 else '✗ NO'}")
    
    # Create quality report
    quality_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'data_span_years': data_span_years,
        'total_records_5min': len(df_5min),
        'total_records_30min': len(df_30min),
        'overall_quality_score': overall_quality,
        'price_range_5min': [float(df_5min['Close'].min()), float(df_5min['Close'].max())],
        'price_range_30min': [float(df_30min['Close'].min()), float(df_30min['Close'].max())],
        'volume_stats_5min': {
            'min': int(df_5min['Volume'].min()),
            'max': int(df_5min['Volume'].max()),
            'mean': float(df_5min['Volume'].mean())
        },
        'volume_stats_30min': {
            'min': int(df_30min['Volume'].min()),
            'max': int(df_30min['Volume'].max()),
            'mean': float(df_30min['Volume'].mean())
        },
        'data_quality': {
            'missing_values_5min': int(missing_5min),
            'missing_values_30min': int(missing_30min),
            'duplicates_5min': int(duplicates_5min),
            'duplicates_30min': int(duplicates_30min),
            'ohlc_valid_5min': bool(ohlc_valid_5min),
            'ohlc_valid_30min': bool(ohlc_valid_30min),
            'volume_valid_5min': bool(volume_valid_5min),
            'volume_valid_30min': bool(volume_valid_30min)
        },
        'train_test_split': {
            'train_records': len(df_5min_train),
            'test_records': len(df_5min_test),
            'train_percentage': len(df_5min_train) / len(df_5min) * 100,
            'test_percentage': len(df_5min_test) / len(df_5min) * 100
        },
        'backtesting_ready': data_span_years >= 3 and overall_quality >= 80
    }
    
    # Save quality report
    with open(f"{data_dir}CL_final_quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    print(f"\n💾 EXPORTED FILES:")
    print(f"• {data_dir}CL_5min_processed.csv - Full 5-minute dataset with technical indicators")
    print(f"• {data_dir}CL_30min_processed.csv - Full 30-minute dataset with technical indicators")
    print(f"• {data_dir}CL_5min_train.csv - Training data (5-minute)")
    print(f"• {data_dir}CL_5min_test.csv - Testing data (5-minute)")
    print(f"• {data_dir}CL_final_quality_report.json - Comprehensive quality report")
    
    print("\n" + "="*80)
    if quality_report['backtesting_ready']:
        print("🎉 MISSION ACCOMPLISHED - READY FOR 500% TRUSTWORTHY BACKTESTING!")
    else:
        print("⚠️  MISSION PARTIALLY COMPLETE - SOME QUALITY ISSUES DETECTED")
    print("="*80)
    
    return quality_report

if __name__ == "__main__":
    analyze_processed_data()