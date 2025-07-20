#!/usr/bin/env python3
"""
Example Usage of the Comprehensive Signal Generation System

This script demonstrates various ways to use the signal generation system
with different configurations and scenarios.
"""

import sys
import os
sys.path.append('/home/QuantNova/GrandModel')

import numpy as np
import pandas as pd
import time
from typing import List, Dict

# Import directly from the signal generation module
from lorentzian_strategy.signal_generation import (
    create_signal_generator,
    create_default_config,
    create_optimized_config,
    SignalType,
    SignalQuality,
    SignalConfig
)

def generate_sample_data(n_bars: int = 1000, base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Create price series with trend and noise
    returns = np.random.normal(0.0001, 0.015, n_bars)
    trend = np.linspace(0, 0.05, n_bars)  # 5% trend
    cycle = 0.02 * np.sin(np.linspace(0, 6*np.pi, n_bars))
    
    log_returns = returns + trend/n_bars + cycle
    prices = base_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC
    volatility = 0.008
    high = prices * (1 + np.abs(np.random.normal(0, volatility, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, volatility, n_bars)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price
    
    # Generate volume
    volume = np.random.lognormal(10, 0.3, n_bars)
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1H'),
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })

def example_basic_usage():
    """Basic signal generation example"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create signal generator with default configuration
    generator = create_signal_generator()
    
    # Generate sample data
    data = generate_sample_data(200)
    print(f"Generated {len(data)} bars of sample data")
    
    # Generate a single signal
    current_price = data['close'].iloc[-1]
    signal = generator.generate_signal(data, current_price)
    
    # Display results
    print(f"\nSignal Generated:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Quality: {signal.signal_quality.value}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Strength: {signal.signal_strength:.3f}")
    print(f"  Current Price: ${signal.current_price:.2f}")
    
    if signal.signal_type != SignalType.NO_SIGNAL:
        print(f"  Entry Price: ${signal.entry_price:.2f}")
        if signal.stop_loss:
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"  Take Profit: ${signal.take_profit:.2f}")
    
    print(f"  ML Prediction: {signal.ml_prediction:.3f}")
    print(f"  Neighbors Found: {signal.neighbors_found}")
    print(f"  Processing Time: {signal.processing_time_ms:.1f}ms")
    print(f"  Market Regime: {signal.regime.value}")
    
    return generator, signal

def example_batch_processing():
    """Batch signal processing example"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create optimized generator for batch processing
    generator = create_signal_generator(create_optimized_config().__dict__)
    
    # Generate longer dataset
    data = generate_sample_data(500)
    
    # Process signals for rolling windows
    signals = []
    processing_times = []
    window_size = 150
    num_signals = 50
    
    print(f"Processing {num_signals} signals with {window_size}-bar windows...")
    
    start_time = time.time()
    
    for i in range(window_size, window_size + num_signals):
        signal_start = time.perf_counter()
        
        window_data = data.iloc[:i+1]
        current_price = window_data['close'].iloc[-1]
        
        signal = generator.generate_signal(window_data, current_price)
        signals.append(signal)
        
        processing_time = (time.perf_counter() - signal_start) * 1000
        processing_times.append(processing_time)
        
        if i % 10 == 0:
            print(f"  Processed {i - window_size + 1}/{num_signals} signals")
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\nBatch Processing Results:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Time per Signal: {np.mean(processing_times):.1f}ms")
    print(f"  95th Percentile Time: {np.percentile(processing_times, 95):.1f}ms")
    print(f"  Max Time: {np.max(processing_times):.1f}ms")
    
    # Signal distribution
    signal_types = {}
    quality_levels = {}
    
    for signal in signals:
        signal_type = signal.signal_type.value
        quality = signal.signal_quality.value
        
        signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        quality_levels[quality] = quality_levels.get(quality, 0) + 1
    
    print(f"\nSignal Distribution:")
    for signal_type, count in signal_types.items():
        percentage = (count / len(signals)) * 100
        print(f"  {signal_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nQuality Distribution:")
    for quality, count in quality_levels.items():
        percentage = (count / len(signals)) * 100
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    return signals, processing_times

def run_examples():
    """Run key examples"""
    print("COMPREHENSIVE SIGNAL GENERATION SYSTEM")
    print("Example Usage Demonstrations")
    print("=" * 80)
    
    try:
        # Run examples
        example_basic_usage()
        example_batch_processing()
        
        print("\n" + "=" * 80)
        print("✅ EXAMPLES COMPLETED SUCCESSFULLY!")
        print("The signal generation system is working correctly.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_examples()