#!/usr/bin/env python3
"""
AGENT 4 - NW-RQK STRATEGY VECTORBT IMPLEMENTATION SPECIALIST

Professional VectorBT Implementation of NW-RQK-based Strategies
Implements NW-RQK → MLMI → FVG and NW-RQK → FVG → MLMI strategies with optimized kernel calculations

Author: AGENT 4 - NW-RQK Strategy VectorBT Implementation Specialist
Date: 2025-07-16
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the source directory to the path
sys.path.append('/home/QuantNova/GrandModel/src')

warnings.filterwarnings('ignore')

# Configure VectorBT for optimal performance
vbt.settings.broadcasting['index_from'] = 'strict'
vbt.settings.caching['enabled'] = True
vbt.settings.returns['year_freq'] = '252D'
vbt.settings.portfolio['init_cash'] = 100000.0
vbt.settings.portfolio['fees'] = 0.001
vbt.settings.portfolio['slippage'] = 0.0005

class OptimizedNWRQKKernel:
    """
    Optimized Nadaraya-Watson Rational Quadratic Kernel Calculator
    Built for maximum performance with vectorbt integration
    """
    
    def __init__(self, h=8.0, r=8.0, alpha=1.0):
        """
        Initialize optimized NW-RQK kernel
        
        Args:
            h: Bandwidth parameter (default: 8.0)
            r: Kernel parameter (default: 8.0)
            alpha: RQ kernel parameter (default: 1.0)
        """
        self.h = h
        self.r = r
        self.alpha = alpha
        self.two_alpha_h_squared = 2.0 * alpha * h * h
        
    def rational_quadratic_kernel(self, x_t, x_i):
        """
        Rational quadratic kernel calculation
        """
        distance_squared = (x_t - x_i) ** 2
        denominator = self.two_alpha_h_squared + distance_squared
        return (self.two_alpha_h_squared / denominator) ** self.alpha
    
    def kernel_regression_batch(self, prices, window_size):
        """
        Batch kernel regression for entire price series
        Optimized for vectorbt performance requirements
        """
        n = len(prices)
        result = np.full(n, np.nan)
        
        # Precompute constants
        two_r_h_squared = 2.0 * self.r * self.h * self.h
        
        for i in range(window_size, n):
            # Get window of historical prices
            window_start = max(0, i - window_size)
            window_prices = prices[window_start:i+1]
            current_price = prices[i]
            
            # Calculate weights
            total_weight = 0.0
            weighted_sum = 0.0
            
            for j in range(len(window_prices)):
                distance_squared = (current_price - window_prices[j]) ** 2
                denominator = two_r_h_squared + distance_squared
                weight = (two_r_h_squared / denominator) ** self.r
                
                weighted_sum += window_prices[j] * weight
                total_weight += weight
            
            # Avoid division by zero
            if total_weight > 0:
                result[i] = weighted_sum / total_weight
                
        return result
    
    def calculate_nwrqk_signals(self, prices: np.ndarray, window_size: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculate NW-RQK signals optimized for vectorbt
        
        Args:
            prices: Price array (numpy array)
            window_size: Lookback window size
            
        Returns:
            Dictionary with NW-RQK values and signals
        """
        # Primary kernel regression
        nwrqk_value = self.kernel_regression_batch(prices, window_size)
        
        # Secondary kernel regression with lag
        nwrqk_lagged = self.kernel_regression_batch(prices, window_size - 5)
        
        # Calculate derivatives and signals
        nwrqk_slope = np.gradient(nwrqk_value)
        nwrqk_acceleration = np.gradient(nwrqk_slope)
        
        # Crossover detection
        bullish_cross = self._detect_crossovers(nwrqk_value, nwrqk_lagged, direction='bullish')
        bearish_cross = self._detect_crossovers(nwrqk_value, nwrqk_lagged, direction='bearish')
        
        # Signal generation
        nwrqk_signal = np.zeros(len(prices))
        nwrqk_signal[bullish_cross] = 1
        nwrqk_signal[bearish_cross] = -1
        
        # Signal strength calculation
        slope_std = np.nanstd(nwrqk_slope)
        if slope_std > 0:
            signal_strength = np.abs(nwrqk_slope) / slope_std
        else:
            signal_strength = np.zeros(len(prices))
        signal_strength = np.nan_to_num(signal_strength, 0)
        
        return {
            'nwrqk_value': nwrqk_value,
            'nwrqk_lagged': nwrqk_lagged,
            'nwrqk_slope': nwrqk_slope,
            'nwrqk_acceleration': nwrqk_acceleration,
            'nwrqk_signal': nwrqk_signal,
            'signal_strength': signal_strength,
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross
        }
    
    def _detect_crossovers(self, series1, series2, direction='bullish'):
        """
        Detect crossovers between two series
        """
        n = len(series1)
        crossovers = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if not (np.isnan(series1[i]) or np.isnan(series2[i]) or 
                    np.isnan(series1[i-1]) or np.isnan(series2[i-1])):
                
                if direction == 'bullish':
                    # Bullish crossover: series1 crosses above series2
                    if series1[i] > series2[i] and series1[i-1] <= series2[i-1]:
                        crossovers[i] = True
                elif direction == 'bearish':
                    # Bearish crossover: series1 crosses below series2
                    if series1[i] < series2[i] and series1[i-1] >= series2[i-1]:
                        crossovers[i] = True
                        
        return crossovers


class IndicatorCalculator:
    """
    Optimized indicator calculations for vectorbt strategies
    """
    
    @staticmethod
    def calculate_rsi(close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI manually"""
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    @staticmethod
    def calculate_macd(close_series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD manually"""
        ema_fast = close_series.ewm(span=fast).mean()
        ema_slow = close_series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line.values, macd_signal.values

    @staticmethod
    def calculate_bollinger_bands(close_series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands manually"""
        sma = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.values, lower_band.values

    @staticmethod
    def calculate_mlmi(df: pd.DataFrame, period: int = 20) -> Dict[str, np.ndarray]:
        """
        Calculate MLMI (Machine Learning Market Index) - Optimized for vectorbt
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        close_series = pd.Series(close)
        
        # RSI calculation
        rsi = IndicatorCalculator.calculate_rsi(close_series, period)
        
        # MACD calculation
        macd_line, macd_signal = IndicatorCalculator.calculate_macd(close_series)
        
        # Bollinger Bands
        bb_upper, bb_lower = IndicatorCalculator.calculate_bollinger_bands(close_series, period)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Volume analysis
        volume_sma = pd.Series(volume).rolling(period).mean().values
        volume_ratio = volume / volume_sma
        
        # Combine into MLMI
        mlmi_components = np.column_stack([
            np.nan_to_num((rsi - 50) / 50, 0),
            np.nan_to_num((macd_line - macd_signal) / close, 0),
            np.nan_to_num(bb_position - 0.5, 0),
            np.nan_to_num(volume_ratio - 1, 0)
        ])
        
        # Fill NaN values
        mlmi_components = np.nan_to_num(mlmi_components, 0)
        
        # Calculate MLMI value
        mlmi_value = 50 + np.mean(mlmi_components, axis=1) * 50
        mlmi_value = np.clip(mlmi_value, 0, 100)
        
        # Generate signals
        mlmi_signal = np.zeros(len(close))
        mlmi_signal[mlmi_value > 65] = 1  # Bullish
        mlmi_signal[mlmi_value < 35] = -1  # Bearish
        
        return {
            'mlmi_value': mlmi_value,
            'mlmi_signal': mlmi_signal,
            'mlmi_strength': np.abs(mlmi_value - 50) / 50,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': macd_signal
        }
    
    @staticmethod
    def calculate_fvg(df: pd.DataFrame, min_gap_size: float = 0.001) -> Dict[str, np.ndarray]:
        """
        Calculate FVG (Fair Value Gap) - Optimized for vectorbt
        """
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        n = len(high)
        fvg_bullish = np.zeros(n, dtype=bool)
        fvg_bearish = np.zeros(n, dtype=bool)
        fvg_signal = np.zeros(n)
        
        # Vectorized FVG detection
        for i in range(2, n):
            # Bullish FVG: current low > previous high (2 bars ago)
            if low[i] > high[i-2]:
                gap_size = (low[i] - high[i-2]) / close[i]
                if gap_size >= min_gap_size:
                    fvg_bullish[i] = True
                    fvg_signal[i] = 1
            
            # Bearish FVG: current high < previous low (2 bars ago)
            elif high[i] < low[i-2]:
                gap_size = (low[i-2] - high[i]) / close[i]
                if gap_size >= min_gap_size:
                    fvg_bearish[i] = True
                    fvg_signal[i] = -1
        
        # Rolling FVG activity
        fvg_activity = pd.Series(fvg_signal).rolling(20).sum().values
        
        return {
            'fvg_signal': fvg_signal,
            'fvg_bullish': fvg_bullish,
            'fvg_bearish': fvg_bearish,
            'fvg_activity': fvg_activity
        }


class NWRQKStrategies:
    """
    NW-RQK-based strategy implementations for vectorbt
    """
    
    def __init__(self, data_path: str = "/home/QuantNova/GrandModel/data/prepared/nq_5m_vectorbt.csv"):
        """
        Initialize NW-RQK strategy framework
        
        Args:
            data_path: Path to historical data file
        """
        self.data_path = Path(data_path)
        self.results_dir = Path('/home/QuantNova/GrandModel/results/nq_backtest')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.nwrqk_kernel = OptimizedNWRQKKernel(h=8.0, r=8.0, alpha=1.0)
        self.indicator_calc = IndicatorCalculator()
        
        # Strategy parameters
        self.lookback_window = 50
        self.signal_threshold = 0.6
        self.min_signal_strength = 0.3
        
    def load_data(self, start_date: str = '2024-01-01', end_date: str = '2024-07-01') -> pd.DataFrame:
        """
        Load NQ data for backtesting
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"🔄 Loading NQ data from {start_date} to {end_date}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        # Standardize column names
        df.columns = [col.title() for col in df.columns]
        
        # Filter date range
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Clean data
        df = df.dropna()
        
        if len(df) == 0:
            print("⚠️ No data found in the specified date range, using all available data")
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            df.columns = [col.title() for col in df.columns]
            df = df.dropna()
        
        print(f"✅ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate all indicators for the strategies
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary of all calculated indicators
        """
        print("🔄 Calculating all indicators...")
        
        indicators = {}
        
        # NW-RQK indicators
        nwrqk_results = self.nwrqk_kernel.calculate_nwrqk_signals(
            df['Close'].values, 
            self.lookback_window
        )
        indicators.update(nwrqk_results)
        
        # MLMI indicators
        mlmi_results = self.indicator_calc.calculate_mlmi(df)
        indicators.update(mlmi_results)
        
        # FVG indicators
        fvg_results = self.indicator_calc.calculate_fvg(df)
        indicators.update(fvg_results)
        
        print("✅ All indicators calculated")
        return indicators
    
    def strategy_nwrqk_mlmi_fvg(self, df: pd.DataFrame, indicators: Dict[str, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        NW-RQK → MLMI → FVG Strategy
        
        Entry Logic:
        1. NW-RQK generates primary signal
        2. MLMI confirms momentum direction
        3. FVG provides final entry timing
        
        Args:
            df: OHLCV DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Tuple of (entries, exits) as pandas Series
        """
        print("🎯 Implementing NW-RQK → MLMI → FVG strategy...")
        
        # Extract signals
        nwrqk_signal = indicators['nwrqk_signal']
        nwrqk_strength = indicators['signal_strength']
        mlmi_signal = indicators['mlmi_signal']
        mlmi_strength = indicators['mlmi_strength']
        fvg_signal = indicators['fvg_signal']
        
        # Initialize entry and exit signals
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        
        # Strategy logic
        for i in range(self.lookback_window, len(df)):
            # Step 1: NW-RQK primary signal
            if nwrqk_signal[i] != 0 and nwrqk_strength[i] > self.min_signal_strength:
                direction = nwrqk_signal[i]
                
                # Step 2: MLMI momentum confirmation
                if mlmi_signal[i] == direction and mlmi_strength[i] > 0.2:
                    
                    # Step 3: FVG entry timing (within 5 bars)
                    fvg_window = fvg_signal[max(0, i-5):i+1]
                    if np.any(fvg_window == direction):
                        entries.iloc[i] = True
        
        # Exit logic: Opposite NW-RQK signal or extreme MLMI
        for i in range(1, len(df)):
            if nwrqk_signal[i] != 0 and i > 0:
                # Check if signal reversed
                if nwrqk_signal[i] * nwrqk_signal[i-1] < 0:
                    exits.iloc[i] = True
            
            # Exit on extreme MLMI levels
            if indicators['mlmi_value'][i] > 85 or indicators['mlmi_value'][i] < 15:
                exits.iloc[i] = True
        
        print(f"✅ Generated {entries.sum()} entries and {exits.sum()} exits")
        return entries, exits
    
    def strategy_nwrqk_fvg_mlmi(self, df: pd.DataFrame, indicators: Dict[str, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        NW-RQK → FVG → MLMI Strategy
        
        Entry Logic:
        1. NW-RQK generates primary signal
        2. FVG provides immediate entry opportunity
        3. MLMI confirms momentum sustainability
        
        Args:
            df: OHLCV DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Tuple of (entries, exits) as pandas Series
        """
        print("🎯 Implementing NW-RQK → FVG → MLMI strategy...")
        
        # Extract signals
        nwrqk_signal = indicators['nwrqk_signal']
        nwrqk_strength = indicators['signal_strength']
        mlmi_signal = indicators['mlmi_signal']
        mlmi_value = indicators['mlmi_value']
        fvg_signal = indicators['fvg_signal']
        
        # Initialize entry and exit signals
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        
        # Strategy logic
        for i in range(self.lookback_window, len(df)):
            # Step 1: NW-RQK primary signal
            if nwrqk_signal[i] != 0 and nwrqk_strength[i] > self.min_signal_strength:
                direction = nwrqk_signal[i]
                
                # Step 2: FVG immediate entry opportunity
                if fvg_signal[i] == direction:
                    
                    # Step 3: MLMI momentum sustainability (within 3 bars)
                    mlmi_window = mlmi_signal[max(0, i-3):i+1]
                    mlmi_value_window = mlmi_value[max(0, i-3):i+1]
                    
                    # Check for MLMI confirmation
                    if direction > 0:  # Bullish
                        if np.any(mlmi_window > 0) or np.any(mlmi_value_window > 55):
                            entries.iloc[i] = True
                    else:  # Bearish
                        if np.any(mlmi_window < 0) or np.any(mlmi_value_window < 45):
                            entries.iloc[i] = True
        
        # Exit logic: NW-RQK signal weakening or MLMI divergence
        for i in range(1, len(df)):
            # Exit on signal weakness
            if nwrqk_strength[i] < 0.1:
                exits.iloc[i] = True
            
            # Exit on MLMI divergence
            if i > 0 and mlmi_signal[i] != 0:
                if mlmi_signal[i] * mlmi_signal[i-1] < 0:
                    exits.iloc[i] = True
        
        print(f"✅ Generated {entries.sum()} entries and {exits.sum()} exits")
        return entries, exits
    
    def run_backtest(self, df: pd.DataFrame, entries: pd.Series, exits: pd.Series, 
                    strategy_name: str) -> vbt.Portfolio:
        """
        Run vectorbt backtest for a strategy
        
        Args:
            df: OHLCV DataFrame
            entries: Entry signals
            exits: Exit signals
            strategy_name: Name of the strategy
            
        Returns:
            VectorBT Portfolio object
        """
        print(f"🔄 Running backtest for {strategy_name}...")
        
        # Portfolio configuration
        portfolio = vbt.Portfolio.from_signals(
            df['Close'],
            entries=entries,
            exits=exits,
            init_cash=100000,
            size=0.95,
            size_type='percent',
            fees=0.001,
            slippage=0.0005,
            freq='5min'
        )
        
        print(f"✅ Backtest completed for {strategy_name}")
        return portfolio
    
    def run_complete_implementation(self):
        """
        Run complete NW-RQK strategies implementation
        
        Returns:
            Tuple of (portfolios, report_file)
        """
        print("🚀 AGENT 4 - NW-RQK VECTORBT STRATEGIES IMPLEMENTATION STARTING")
        print("="*70)
        
        # Load data
        df = self.load_data()
        
        # Calculate indicators
        indicators = self.calculate_all_indicators(df)
        
        # Strategy 1: NW-RQK → MLMI → FVG
        entries1, exits1 = self.strategy_nwrqk_mlmi_fvg(df, indicators)
        portfolio1 = self.run_backtest(df, entries1, exits1, "NW-RQK → MLMI → FVG")
        
        # Strategy 2: NW-RQK → FVG → MLMI
        entries2, exits2 = self.strategy_nwrqk_fvg_mlmi(df, indicators)
        portfolio2 = self.run_backtest(df, entries2, exits2, "NW-RQK → FVG → MLMI")
        
        # Basic performance analysis
        print("\n🎯 PERFORMANCE RESULTS:")
        print("="*50)
        
        # Strategy 1 results
        stats1 = portfolio1.stats()
        print(f"\n📊 NW-RQK → MLMI → FVG Strategy:")
        print(f"   Total Return: {stats1.get('Total Return [%]', 0):.2f}%")
        print(f"   Sharpe Ratio: {stats1.get('Sharpe Ratio', 0):.2f}")
        print(f"   Win Rate: {stats1.get('Win Rate [%]', 0):.2f}%")
        print(f"   Total Trades: {stats1.get('# Trades', 0)}")
        print(f"   Max Drawdown: {stats1.get('Max Drawdown [%]', 0):.2f}%")
        
        # Strategy 2 results
        stats2 = portfolio2.stats()
        print(f"\n📊 NW-RQK → FVG → MLMI Strategy:")
        print(f"   Total Return: {stats2.get('Total Return [%]', 0):.2f}%")
        print(f"   Sharpe Ratio: {stats2.get('Sharpe Ratio', 0):.2f}")
        print(f"   Win Rate: {stats2.get('Win Rate [%]', 0):.2f}%")
        print(f"   Total Trades: {stats2.get('# Trades', 0)}")
        print(f"   Max Drawdown: {stats2.get('Max Drawdown [%]', 0):.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'agent_info': {
                'agent_name': 'AGENT 4 - NW-RQK Strategy VectorBT Implementation Specialist',
                'mission': 'Implement NW-RQK-based strategies in vectorbt for professional backtesting',
                'timestamp': timestamp
            },
            'strategy_1': {
                'name': 'NW-RQK → MLMI → FVG',
                'total_return_pct': float(stats1.get('Total Return [%]', 0)),
                'sharpe_ratio': float(stats1.get('Sharpe Ratio', 0)),
                'win_rate_pct': float(stats1.get('Win Rate [%]', 0)),
                'total_trades': int(stats1.get('# Trades', 0)),
                'max_drawdown_pct': float(stats1.get('Max Drawdown [%]', 0))
            },
            'strategy_2': {
                'name': 'NW-RQK → FVG → MLMI',
                'total_return_pct': float(stats2.get('Total Return [%]', 0)),
                'sharpe_ratio': float(stats2.get('Sharpe Ratio', 0)),
                'win_rate_pct': float(stats2.get('Win Rate [%]', 0)),
                'total_trades': int(stats2.get('# Trades', 0)),
                'max_drawdown_pct': float(stats2.get('Max Drawdown [%]', 0))
            }
        }
        
        report_file = self.results_dir / f'nwrqk_vectorbt_strategies_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 AGENT 4 MISSION COMPLETE!")
        print(f"📄 Report saved: {report_file}")
        
        return {
            'portfolios': {'strategy1': portfolio1, 'strategy2': portfolio2},
            'results': results,
            'report_file': str(report_file)
        }


def main():
    """
    Main function to run NW-RQK strategies implementation
    """
    print("🎯 AGENT 4 - NW-RQK STRATEGY VECTORBT IMPLEMENTATION SPECIALIST")
    print("Professional implementation of NW-RQK-based strategies using vectorbt")
    print("Strategies: NW-RQK → MLMI → FVG and NW-RQK → FVG → MLMI")
    print()
    
    # Initialize and run implementation
    strategies = NWRQKStrategies()
    results = strategies.run_complete_implementation()
    
    return results


if __name__ == "__main__":
    results = main()