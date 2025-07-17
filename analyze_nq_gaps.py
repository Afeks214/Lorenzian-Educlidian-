#!/usr/bin/env python3
"""
Analyze actual NQ price data to determine realistic FVG parameters
"""

import pandas as pd
import numpy as np

def analyze_nq_gaps():
    """Analyze NQ price data to find realistic FVG parameters"""
    
    # Load data
    file_path = "/home/QuantNova/GrandModel/colab/data/@NQ - 5 min - ETH.csv"
    df = pd.read_csv(file_path)
    
    # Take a sample for analysis
    sample_df = df.head(10000).copy()
    
    print(f"📊 Analyzing {len(sample_df)} bars of NQ data...")
    print(f"Price range: ${sample_df['Low'].min():.2f} - ${sample_df['High'].max():.2f}")
    
    # Analyze potential gaps
    gaps_found = []
    
    for i in range(2, len(sample_df)):
        candle1_high = sample_df['High'].iloc[i-2]
        candle1_low = sample_df['Low'].iloc[i-2]
        candle3_high = sample_df['High'].iloc[i]
        candle3_low = sample_df['Low'].iloc[i]
        current_price = sample_df['Close'].iloc[i]
        
        # Check for bullish gaps
        if candle3_low > candle1_high:
            gap_size = candle3_low - candle1_high
            gap_percent = gap_size / current_price * 100
            gaps_found.append(('bull', gap_size, gap_percent, i))
        
        # Check for bearish gaps
        elif candle3_high < candle1_low:
            gap_size = candle1_low - candle3_high
            gap_percent = gap_size / current_price * 100
            gaps_found.append(('bear', gap_size, gap_percent, i))
    
    if gaps_found:
        gap_sizes = [g[1] for g in gaps_found]
        gap_percents = [g[2] for g in gaps_found]
        
        print(f"\n📈 Found {len(gaps_found)} potential gaps:")
        print(f"   Gap sizes: min={min(gap_sizes):.2f}, max={max(gap_sizes):.2f}, avg={np.mean(gap_sizes):.2f}")
        print(f"   Gap %: min={min(gap_percents):.4f}%, max={max(gap_percents):.4f}%, avg={np.mean(gap_percents):.4f}%")
        
        # Show distribution
        print(f"\n📊 Gap size distribution:")
        for threshold in [0.25, 0.5, 1.0, 2.0, 4.0]:
            count = sum(1 for g in gap_sizes if g >= threshold)
            percent = count / len(gaps_found) * 100
            print(f"   >= {threshold:.2f} points: {count} gaps ({percent:.1f}%)")
        
        # Show some examples
        print(f"\n📋 First 10 gap examples:")
        for i, (gap_type, size, percent, bar_idx) in enumerate(gaps_found[:10]):
            print(f"   {i+1}. {gap_type.upper()} gap at bar {bar_idx}: {size:.2f} points ({percent:.4f}%)")
        
        # Recommend parameters
        avg_size = np.mean(gap_sizes)
        median_size = np.median(gap_sizes)
        avg_percent = np.mean(gap_percents)
        
        print(f"\n🎯 RECOMMENDED PARAMETERS:")
        print(f"   min_gap_ticks: {median_size:.2f} (median gap size)")
        print(f"   min_gap_percent: {avg_percent/100:.6f} ({avg_percent:.4f}%)")
        print(f"   For conservative detection: min_gap_ticks={median_size * 1.5:.2f}")
        
    else:
        print("❌ No gaps found with current logic - check implementation")

if __name__ == "__main__":
    analyze_nq_gaps()