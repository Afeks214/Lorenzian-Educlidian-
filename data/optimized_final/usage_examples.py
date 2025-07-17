#!/usr/bin/env python3
"""
USAGE EXAMPLES FOR OPTIMIZED NQ DATA
Generated on 2025-07-17 09:03:14
"""

import pandas as pd
import numpy as np
import time

# File paths
OHLCV_CSV = "/home/QuantNova/GrandModel/data/optimized_final/NQ_30m_OHLCV_optimized.csv"
OHLCV_PICKLE = "/home/QuantNova/GrandModel/data/optimized_final/NQ_30m_OHLCV_optimized.pkl"
RETURNS_CSV = "/home/QuantNova/GrandModel/data/optimized_final/NQ_30m_returns_indicators.csv"
RETURNS_PICKLE = "/home/QuantNova/GrandModel/data/optimized_final/NQ_30m_returns_indicators.pkl"
NUMPY_ARRAYS = "/home/QuantNova/GrandModel/data/optimized_final/NQ_30m_numpy_arrays.npz"

def example_1_basic_loading():
    """Example 1: Basic data loading for backtesting"""
    print("Example 1: Basic OHLCV loading")
    
    # Method 1: CSV (slower but widely compatible)
    start = time.time()
    df = pd.read_csv(OHLCV_CSV, parse_dates=['Timestamp'])
    print(f"CSV load time: {time.time() - start:.3f}s")
    
    # Method 2: Pickle (faster)
    start = time.time()
    df = pd.read_pickle(OHLCV_PICKLE)
    print(f"Pickle load time: {time.time() - start:.3f}s")
    
    print(f"Data shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
    
    return df

def example_2_numpy_arrays():
    """Example 2: Ultra-fast numpy arrays for algorithms"""
    print("\nExample 2: Numpy arrays for high-performance")
    
    start = time.time()
    data = np.load(NUMPY_ARRAYS)
    ohlcv = data['ohlcv']
    timestamps = data['timestamps']
    
    print(f"Numpy load time: {time.time() - start:.3f}s")
    print(f"OHLCV shape: {ohlcv.shape}")
    print(f"Data types: {ohlcv.dtype}")
    
    # Access specific columns
    opens = ohlcv[:, 0]
    highs = ohlcv[:, 1]
    lows = ohlcv[:, 2]
    closes = ohlcv[:, 3]
    volumes = ohlcv[:, 4]
    
    # Fast calculations
    returns = np.diff(np.log(closes))
    volatility = np.std(returns)
    
    print(f"Fast calculations completed in microseconds")
    
    return ohlcv, timestamps

def example_3_returns_and_indicators():
    """Example 3: Pre-calculated returns and indicators"""
    print("\nExample 3: Returns and technical indicators")
    
    start = time.time()
    df = pd.read_pickle(RETURNS_PICKLE)
    print(f"Load time: {time.time() - start:.3f}s")
    
    print(f"Available columns: {df.columns.tolist()}")
    
    # Ready-to-use indicators
    signal = np.where(df['sma_20'] > df['sma_50'], 1, -1)
    strategy_returns = signal * df['returns'].shift(-1)
    
    print(f"Strategy performance ready for analysis")
    
    return df

def example_4_backtesting_simulation():
    """Example 4: Simple backtesting simulation"""
    print("\nExample 4: Backtesting simulation")
    
    # Load data
    df = pd.read_pickle(OHLCV_PICKLE)
    
    # Calculate indicators
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    
    # Generate signals
    df['signal'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Performance metrics
    total_return = df['strategy_returns'].sum()
    volatility = df['strategy_returns'].std()
    sharpe_ratio = df['strategy_returns'].mean() / volatility if volatility > 0 else 0
    
    print(f"Total return: {total_return:.2%}")
    print(f"Volatility: {volatility:.4f}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    
    return df

if __name__ == "__main__":
    print("🚀 NQ Data Usage Examples")
    print("=" * 50)
    
    # Run examples
    df1 = example_1_basic_loading()
    ohlcv, timestamps = example_2_numpy_arrays()
    df3 = example_3_returns_and_indicators()
    df4 = example_4_backtesting_simulation()
    
    print("\n✅ All examples completed successfully!")
