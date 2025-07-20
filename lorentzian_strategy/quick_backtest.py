#!/usr/bin/env python3
"""
Quick Lorentzian Strategy Backtest
==================================

Simple backtest to demonstrate the Lorentzian strategy performance
without complex dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """Load and clean the NQ data with flexible timestamp parsing"""
    print("📊 Loading NQ 30-minute data...")
    
    try:
        # Read with flexible timestamp parsing
        df = pd.read_csv(file_path)
        
        # Clean and standardize timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
        
        # Sort by timestamp
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Basic validation
        print(f"✅ Data loaded: {len(df):,} records")
        print(f"📅 Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"💰 Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for Lorentzian features"""
    print("🔧 Calculating technical indicators...")
    
    # Simple moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    df['Returns'] = df['Close'].pct_change()
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_20'] = df['Close'].pct_change(20)
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    print("✅ Technical indicators calculated")
    return df

def generate_lorentzian_signals(df):
    """Generate trading signals using simplified Lorentzian logic"""
    print("🎯 Generating Lorentzian-inspired trading signals...")
    
    # Normalize features for distance calculation
    features = ['RSI', 'Momentum_5', 'Momentum_20', 'Volatility', 'Volume_Ratio']
    
    # Fill NaN values
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].median())
    
    # Normalize features to [0, 1]
    for feature in features:
        df[f'{feature}_norm'] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    
    # Simple signal generation based on feature combinations
    df['Signal_Score'] = 0
    
    # RSI signals (oversold/overbought)
    df.loc[df['RSI'] < 30, 'Signal_Score'] += 1  # Oversold = buy signal
    df.loc[df['RSI'] > 70, 'Signal_Score'] -= 1  # Overbought = sell signal
    
    # Momentum signals
    df.loc[df['Momentum_5'] > 0.01, 'Signal_Score'] += 1  # Strong positive momentum
    df.loc[df['Momentum_5'] < -0.01, 'Signal_Score'] -= 1  # Strong negative momentum
    
    # Trend signals
    df.loc[df['Close'] > df['SMA_20'], 'Signal_Score'] += 0.5  # Above short-term MA
    df.loc[df['Close'] < df['SMA_20'], 'Signal_Score'] -= 0.5  # Below short-term MA
    
    # Generate discrete signals
    df['Signal'] = 0
    df.loc[df['Signal_Score'] >= 1.5, 'Signal'] = 1   # Buy signal
    df.loc[df['Signal_Score'] <= -1.5, 'Signal'] = -1  # Sell signal
    
    # Count signals
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()
    
    print(f"✅ Signals generated: {buy_signals} buy, {sell_signals} sell")
    return df

def run_simple_backtest(df, initial_capital=100000):
    """Run a simple backtest with the generated signals"""
    print("🚀 Running backtest simulation...")
    
    capital = initial_capital
    position = 0
    trades = []
    portfolio_values = [initial_capital]
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['Close']
        signal = df.iloc[i]['Signal']
        
        if signal == 1 and position <= 0:  # Buy signal
            # Calculate position size (50% of capital)
            position_value = capital * 0.5
            shares = position_value / current_price
            position += shares
            capital -= position_value
            trades.append({
                'timestamp': df.iloc[i]['Timestamp'],
                'type': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': position_value
            })
            
        elif signal == -1 and position > 0:  # Sell signal
            # Sell all position
            sell_value = position * current_price
            capital += sell_value
            trades.append({
                'timestamp': df.iloc[i]['Timestamp'],
                'type': 'SELL',
                'price': current_price,
                'shares': position,
                'value': sell_value
            })
            position = 0
        
        # Calculate portfolio value
        portfolio_value = capital + (position * current_price)
        portfolio_values.append(portfolio_value)
    
    # Final liquidation if holding position
    if position > 0:
        final_value = position * df.iloc[-1]['Close']
        capital += final_value
        portfolio_value = capital
    else:
        portfolio_value = capital + (position * df.iloc[-1]['Close'])
    
    return {
        'final_capital': capital,
        'final_portfolio_value': portfolio_value,
        'total_return': (portfolio_value - initial_capital) / initial_capital,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def calculate_performance_metrics(results, df):
    """Calculate comprehensive performance metrics"""
    print("📊 Calculating performance metrics...")
    
    portfolio_values = np.array(results['portfolio_values'])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Basic metrics
    total_return = results['total_return']
    annualized_return = (1 + total_return) ** (252*48 / len(df)) - 1  # 30-min bars
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252*48)  # Annualized
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Trade analysis
    trades = results['trades']
    num_trades = len(trades)
    
    if num_trades >= 2:
        # Calculate trade returns
        trade_returns = []
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy_price = trades[i]['price']
                sell_price = trades[i + 1]['price']
                trade_return = (sell_price - buy_price) / buy_price
                trade_returns.append(trade_return)
        
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
    else:
        win_rate = 0
        profit_factor = 0
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Number of Trades': num_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
    }
    
    return metrics

def main():
    """Main execution function"""
    print("🎯 LORENTZIAN STRATEGY QUICK BACKTEST")
    print("=" * 50)
    
    # File path
    data_path = Path("/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load and prepare data
    df = load_and_clean_data(data_path)
    if df is None:
        return
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Generate signals
    df = generate_lorentzian_signals(df)
    
    # Run backtest
    results = run_simple_backtest(df)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results, df)
    
    # Display results
    print("\n🏆 BACKTEST RESULTS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value}")
    
    print(f"\nInitial Capital    : ${100000:,.2f}")
    print(f"Final Portfolio    : ${results['final_portfolio_value']:,.2f}")
    print(f"Total Profit/Loss  : ${results['final_portfolio_value'] - 100000:,.2f}")
    
    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("=" * 50)
    
    sharpe = float(metrics['Sharpe Ratio'])
    max_dd = float(metrics['Maximum Drawdown'].rstrip('%')) / 100
    win_rate = float(metrics['Win Rate'].rstrip('%')) / 100
    
    passed_tests = 0
    total_tests = 4
    
    print(f"Sharpe Ratio > 1.0     : {'✅ PASS' if sharpe > 1.0 else '❌ FAIL'} ({sharpe:.2f})")
    if sharpe > 1.0: passed_tests += 1
    
    print(f"Max Drawdown < 20%     : {'✅ PASS' if abs(max_dd) < 0.20 else '❌ FAIL'} ({max_dd:.1%})")
    if abs(max_dd) < 0.20: passed_tests += 1
    
    print(f"Win Rate > 40%         : {'✅ PASS' if win_rate > 0.40 else '❌ FAIL'} ({win_rate:.1%})")
    if win_rate > 0.40: passed_tests += 1
    
    total_return = results['total_return']
    print(f"Positive Returns       : {'✅ PASS' if total_return > 0 else '❌ FAIL'} ({total_return:.1%})")
    if total_return > 0: passed_tests += 1
    
    print(f"\nOverall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:
        print("🎉 STRATEGY PERFORMANCE: GOOD")
    elif passed_tests >= 2:
        print("⚠️ STRATEGY PERFORMANCE: FAIR")
    else:
        print("🔴 STRATEGY PERFORMANCE: NEEDS IMPROVEMENT")
    
    print("\n✅ Backtest completed successfully!")

if __name__ == "__main__":
    main()