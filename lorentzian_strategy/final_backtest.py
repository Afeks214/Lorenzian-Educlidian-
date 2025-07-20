#!/usr/bin/env python3
"""
Final Lorentzian Strategy Backtest
==================================

Production-ready backtest with proper Lorentzian implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load and clean the NQ data"""
    print("📊 Loading NQ 30-minute data...")
    
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df):,} records")
    print(f"📅 Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"💰 Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    print("🔧 Calculating technical indicators...")
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Price momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(20).std()
    
    # Volume indicators  
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    print("✅ Indicators calculated")
    return df

def normalize_features(df):
    """Normalize features for distance calculation"""
    features = ['RSI', 'Momentum_5', 'Momentum_10', 'Volatility', 'Volume_Ratio']
    
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].median())
        min_val = df[feature].min()
        max_val = df[feature].max()
        if max_val > min_val:
            df[f'{feature}_norm'] = (df[feature] - min_val) / (max_val - min_val)
        else:
            df[f'{feature}_norm'] = 0.5
    
    return df, features

def generate_signals(df, features, lookback=100, k=8):
    """Generate signals using Lorentzian k-NN"""
    print("🎯 Generating Lorentzian signals...")
    
    norm_features = [f'{f}_norm' for f in features]
    signals = []
    
    for i in range(lookback, len(df)):
        current = df.iloc[i][norm_features].values
        
        distances = []
        labels = []
        
        # Calculate distances to historical points
        for j in range(i - lookback, i):
            historical = df.iloc[j][norm_features].values
            
            # Lorentzian distance with proper numpy handling
            diff = np.abs(current - historical)
            distance = np.sum(np.log1p(diff))  # log1p is log(1 + x)
            distances.append(distance)
            
            # Future return label
            if j + 3 < len(df):
                future_return = (df.iloc[j + 3]['Close'] - df.iloc[j]['Close']) / df.iloc[j]['Close']
                label = 1 if future_return > 0.003 else (-1 if future_return < -0.003 else 0)
            else:
                label = 0
            labels.append(label)
        
        # Find k nearest neighbors
        sorted_pairs = sorted(zip(distances, labels))
        k_nearest = sorted_pairs[:k]
        
        # Weighted voting
        total_weight = 0
        weighted_sum = 0
        
        for dist, label in k_nearest:
            weight = 1 / (1 + dist)
            weighted_sum += weight * label
            total_weight += weight
        
        prediction = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Generate signal
        if prediction > 0.2:
            signal = 1
        elif prediction < -0.2:
            signal = -1
        else:
            signal = 0
            
        signals.append(signal)
    
    # Add signals to dataframe
    df['Signal'] = 0
    df.iloc[lookback:, df.columns.get_loc('Signal')] = signals
    
    # Apply filters
    df = apply_filters(df)
    
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()
    print(f"✅ Signals: {buy_signals} buy, {sell_signals} sell")
    
    return df

def apply_filters(df):
    """Apply trend and quality filters"""
    # Trend filter
    trend_up = df['Close'] > df['SMA_50']
    
    # Volatility filter  
    vol_median = df['Volatility'].median()
    vol_ok = (df['Volatility'] > vol_median * 0.5) & (df['Volatility'] < vol_median * 2)
    
    # Apply filters
    df.loc[(df['Signal'] == 1) & (~trend_up), 'Signal'] = 0
    df.loc[(df['Signal'] == -1) & (trend_up), 'Signal'] = 0
    df.loc[~vol_ok, 'Signal'] = 0
    
    return df

def run_backtest(df, capital=100000):
    """Run backtest with risk management"""
    print("🚀 Running backtest...")
    
    position = 0
    trades = []
    portfolio_values = [capital]
    
    # Risk parameters
    position_size = 0.2  # 20% per trade
    stop_loss = 0.02     # 2% stop
    take_profit = 0.04   # 4% profit target
    
    entry_price = 0
    
    for i in range(1, len(df)):
        price = df.iloc[i]['Close']
        signal = df.iloc[i]['Signal']
        
        # Check stop loss / take profit
        if position > 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                # Exit position
                sell_value = position * price
                capital += sell_value
                trades.append({
                    'type': 'SELL',
                    'price': price,
                    'shares': position,
                    'pnl_pct': pnl_pct
                })
                position = 0
                entry_price = 0
        
        # New signals
        if signal == 1 and position == 0:  # Buy
            invest_amount = capital * position_size
            if invest_amount > 1000:
                shares = invest_amount / price
                position = shares
                capital -= invest_amount
                entry_price = price
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'pnl_pct': 0
                })
        
        elif signal == -1 and position > 0:  # Sell
            sell_value = position * price
            capital += sell_value
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
            trades.append({
                'type': 'SELL',
                'price': price,
                'shares': position,
                'pnl_pct': pnl_pct
            })
            position = 0
            entry_price = 0
        
        # Update portfolio value
        portfolio_value = capital + (position * price)
        portfolio_values.append(portfolio_value)
    
    # Final position
    if position > 0:
        final_value = position * df.iloc[-1]['Close']
        capital += final_value
        
    final_portfolio = capital
    
    return {
        'final_value': final_portfolio,
        'total_return': (final_portfolio - 100000) / 100000,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def calculate_metrics(results, df):
    """Calculate performance metrics"""
    print("📊 Calculating metrics...")
    
    portfolio_values = np.array(results['portfolio_values'])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Basic metrics
    total_return = results['total_return']
    annual_return = (1 + total_return) ** (252*48/len(df)) - 1
    
    volatility = np.std(returns) * np.sqrt(252*48)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Trade analysis
    trades = results['trades']
    trade_returns = [t['pnl_pct'] for t in trades if t['type'] == 'SELL']
    
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
        'Total Trades': len([t for t in trades if t['type'] == 'BUY'])
    }

def main():
    """Main execution"""
    print("🎯 LORENTZIAN STRATEGY BACKTEST RESULTS")
    print("=" * 50)
    
    # Load data
    data_path = Path("/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv")
    df = load_data(data_path)
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Normalize features
    df, features = normalize_features(df)
    
    # Generate signals
    df = generate_signals(df, features)
    
    # Run backtest
    results = run_backtest(df)
    
    # Calculate metrics
    metrics = calculate_metrics(results, df)
    
    # Display results
    print("\n🏆 PERFORMANCE RESULTS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:15}: {value}")
    
    print(f"\nInitial Capital: ${100000:,.2f}")
    print(f"Final Value    : ${results['final_value']:,.2f}")
    print(f"Profit/Loss    : ${results['final_value'] - 100000:,.2f}")
    
    # Target assessment
    print("\n🎯 TARGET ASSESSMENT")
    print("=" * 50)
    
    sharpe = float(metrics['Sharpe Ratio'])
    max_dd = abs(float(metrics['Max Drawdown'].rstrip('%'))) / 100
    win_rate = float(metrics['Win Rate'].rstrip('%')) / 100
    total_ret = results['total_return']
    
    tests = [
        ("Sharpe > 1.5", sharpe > 1.5, f"{sharpe:.2f}"),
        ("Drawdown < 20%", max_dd < 0.20, f"{max_dd:.1%}"),
        ("Win Rate > 50%", win_rate > 0.50, f"{win_rate:.1%}"),
        ("Positive Return", total_ret > 0, f"{total_ret:.1%}")
    ]
    
    passed = 0
    for test_name, result, value in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status} ({value})")
        if result:
            passed += 1
    
    print(f"\nScore: {passed}/4 targets achieved")
    
    if passed >= 3:
        print("🎉 EXCELLENT PERFORMANCE!")
    elif passed >= 2:
        print("🚀 GOOD PERFORMANCE!")
    else:
        print("⚠️ NEEDS OPTIMIZATION")
    
    print("\n✅ Backtest completed!")

if __name__ == "__main__":
    main()