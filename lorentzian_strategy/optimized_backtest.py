#!/usr/bin/env python3
"""
Optimized Lorentzian Strategy Backtest
=====================================

Enhanced backtest with optimized parameters and advanced signal generation
to demonstrate the full potential of the Lorentzian strategy.
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
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"✅ Data loaded: {len(df):,} records")
        print(f"📅 Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"💰 Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def calculate_advanced_indicators(df):
    """Calculate advanced technical indicators optimized for Lorentzian"""
    print("🔧 Calculating advanced technical indicators...")
    
    # Multiple timeframe SMAs
    for period in [10, 20, 50, 100]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
    
    # Advanced RSI with multiple periods
    for period in [14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Wave Trend calculation (simplified)
    esa = df['Close'].ewm(span=10).mean()
    d = (df['High'] + df['Low'] + df['Close']) / 3 - esa
    d_abs_ema = d.abs().ewm(span=10).mean()
    ci = d / (0.015 * d_abs_ema)
    df['WaveTrend1'] = ci.ewm(span=21).mean()
    df['WaveTrend2'] = df['WaveTrend1'].rolling(4).mean()
    
    # CCI calculation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # ADX calculation (simplified)
    high_diff = df['High'] - df['High'].shift(1)
    low_diff = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = np.maximum(df['High'] - df['Low'], 
                   np.maximum(abs(df['High'] - df['Close'].shift(1)),
                             abs(df['Low'] - df['Close'].shift(1))))
    
    plus_dm_14 = pd.Series(plus_dm).rolling(14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(14).sum()
    tr_14 = pd.Series(tr).rolling(14).sum()
    
    plus_di = 100 * (plus_dm_14 / tr_14)
    minus_di = 100 * (minus_dm_14 / tr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()
    
    # Enhanced momentum indicators
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'].pct_change(period)
        
    # Volatility measures
    df['ATR'] = tr.rolling(14).mean()
    df['Volatility_10'] = df['Close'].rolling(10).std()
    df['Volatility_20'] = df['Close'].rolling(20).std()
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['Volume_Momentum'] = df['Volume'].pct_change(5)
    
    print("✅ Advanced technical indicators calculated")
    return df

def calculate_lorentzian_features(df):
    """Calculate normalized features for Lorentzian distance calculation"""
    print("🧮 Calculating Lorentzian features...")
    
    # Select key features for Lorentzian classifier
    feature_columns = [
        'RSI_14', 'RSI_21', 'WaveTrend1', 'WaveTrend2', 'CCI', 'ADX',
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Volatility_10', 'Volatility_20', 'Volume_Ratio'
    ]
    
    # Fill NaN values with median
    for col in feature_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Normalize features to [0, 1] for distance calculation
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f'{col}_norm'] = 0.5
    
    return df, feature_columns

def generate_lorentzian_signals(df, feature_columns, lookback=100, k_neighbors=8):
    """Generate trading signals using Lorentzian k-NN classification"""
    print("🎯 Generating Lorentzian k-NN trading signals...")
    
    normalized_features = [f'{col}_norm' for col in feature_columns]
    
    signals = []
    
    for i in range(lookback, len(df)):
        current_features = df.iloc[i][normalized_features].values
        
        # Calculate Lorentzian distances to historical points
        distances = []
        labels = []
        
        for j in range(i - lookback, i):
            historical_features = df.iloc[j][normalized_features].values
            
            # Lorentzian distance: sum of log(1 + |x_i - y_i|)
            distance = np.sum(np.log(1 + np.abs(current_features - historical_features)))
            distances.append(distance)
            
            # Create label based on future price movement
            if j + 5 < len(df):  # Look ahead 5 periods
                future_return = (df.iloc[j + 5]['Close'] - df.iloc[j]['Close']) / df.iloc[j]['Close']
                label = 1 if future_return > 0.005 else (-1 if future_return < -0.005 else 0)  # 0.5% threshold
            else:
                label = 0
            labels.append(label)
        
        # Find k nearest neighbors
        distance_label_pairs = list(zip(distances, labels))
        distance_label_pairs.sort(key=lambda x: x[0])  # Sort by distance
        
        k_nearest = distance_label_pairs[:k_neighbors]
        k_labels = [pair[1] for pair in k_nearest]
        
        # Weighted voting (closer neighbors have more weight)
        if len(k_labels) > 0:
            weights = [1 / (1 + dist) for dist, _ in k_nearest]  # Inverse distance weighting
            weighted_sum = sum(w * l for w, l in zip(weights, k_labels))
            weight_sum = sum(weights)
            
            if weight_sum > 0:
                prediction = weighted_sum / weight_sum
            else:
                prediction = 0
        else:
            prediction = 0
        
        # Generate signal based on prediction strength
        if prediction > 0.3:
            signal = 1  # Buy
        elif prediction < -0.3:
            signal = -1  # Sell
        else:
            signal = 0  # Hold
        
        signals.append(signal)
    
    # Add signals to dataframe
    df['Signal'] = 0
    df.iloc[lookback:lookback + len(signals), df.columns.get_loc('Signal')] = signals
    
    # Filter signals using trend and volatility
    df = add_filters(df)
    
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()
    
    print(f"✅ Lorentzian signals generated: {buy_signals} buy, {sell_signals} sell")
    return df

def add_filters(df):
    """Add trend and volatility filters to improve signal quality"""
    print("🔍 Applying signal filters...")
    
    # Trend filter: only trade in direction of longer-term trend
    df['Trend_Filter'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
    
    # Volatility filter: avoid trading in very low or very high volatility
    vol_percentile_20 = df['Volatility_20'].quantile(0.2)
    vol_percentile_80 = df['Volatility_20'].quantile(0.8)
    df['Vol_Filter'] = np.where(
        (df['Volatility_20'] > vol_percentile_20) & (df['Volatility_20'] < vol_percentile_80), 
        1, 0
    )
    
    # ADX filter: only trade when trend strength is sufficient
    df['ADX_Filter'] = np.where(df['ADX'] > 20, 1, 0)
    
    # Apply filters
    original_signals = df['Signal'].copy()
    
    # Only allow buy signals when uptrend
    df.loc[(df['Signal'] == 1) & (df['Trend_Filter'] != 1), 'Signal'] = 0
    
    # Only allow sell signals when downtrend  
    df.loc[(df['Signal'] == -1) & (df['Trend_Filter'] != -1), 'Signal'] = 0
    
    # Disable signals in extreme volatility or low ADX
    df.loc[(df['Vol_Filter'] == 0) | (df['ADX_Filter'] == 0), 'Signal'] = 0
    
    filtered_signals = (df['Signal'] != 0).sum()
    original_count = (original_signals != 0).sum()
    
    print(f"✅ Signals after filtering: {filtered_signals} (was {original_count})")
    return df

def run_optimized_backtest(df, initial_capital=100000):
    """Run optimized backtest with dynamic position sizing and risk management"""
    print("🚀 Running optimized backtest with risk management...")
    
    capital = initial_capital
    position = 0
    trades = []
    portfolio_values = [initial_capital]
    max_portfolio_value = initial_capital
    
    # Risk management parameters
    max_position_pct = 0.25  # Max 25% per position
    stop_loss_pct = 0.015    # 1.5% stop loss
    take_profit_pct = 0.03   # 3% take profit
    max_drawdown_stop = 0.15 # Stop trading at 15% drawdown
    
    position_entry_price = 0
    trading_enabled = True
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['Close']
        signal = df.iloc[i]['Signal']
        
        # Update max portfolio value
        current_portfolio_value = capital + (position * current_price)
        max_portfolio_value = max(max_portfolio_value, current_portfolio_value)
        
        # Check drawdown stop
        current_drawdown = (current_portfolio_value - max_portfolio_value) / max_portfolio_value
        if current_drawdown < -max_drawdown_stop:
            trading_enabled = False
        
        # Risk management: check stop loss and take profit
        if position > 0 and position_entry_price > 0:
            # Long position
            price_change = (current_price - position_entry_price) / position_entry_price
            
            if price_change <= -stop_loss_pct or price_change >= take_profit_pct:
                # Exit position (stop loss or take profit)
                sell_value = position * current_price
                capital += sell_value
                trades.append({
                    'timestamp': df.iloc[i]['Timestamp'],
                    'type': 'SELL (Risk Mgmt)',
                    'price': current_price,
                    'shares': position,
                    'value': sell_value,
                    'reason': 'Stop Loss' if price_change <= -stop_loss_pct else 'Take Profit'
                })
                position = 0
                position_entry_price = 0
        
        elif position < 0 and position_entry_price > 0:
            # Short position
            price_change = (position_entry_price - current_price) / position_entry_price
            
            if price_change <= -stop_loss_pct or price_change >= take_profit_pct:
                # Exit position (stop loss or take profit)
                cover_value = abs(position) * current_price
                capital -= cover_value
                trades.append({
                    'timestamp': df.iloc[i]['Timestamp'],
                    'type': 'COVER (Risk Mgmt)',
                    'price': current_price,
                    'shares': abs(position),
                    'value': cover_value,
                    'reason': 'Stop Loss' if price_change <= -stop_loss_pct else 'Take Profit'
                })
                position = 0
                position_entry_price = 0
        
        # Process new signals
        if trading_enabled and signal != 0 and position == 0:
            if signal == 1:  # Buy signal
                position_value = capital * max_position_pct
                if position_value > 1000:  # Minimum position size
                    shares = position_value / current_price
                    position = shares
                    capital -= position_value
                    position_entry_price = current_price
                    trades.append({
                        'timestamp': df.iloc[i]['Timestamp'],
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': position_value,
                        'reason': 'Lorentzian Signal'
                    })
                    
            elif signal == -1:  # Sell (short) signal
                position_value = capital * max_position_pct
                if position_value > 1000:  # Minimum position size
                    shares = position_value / current_price
                    position = -shares  # Negative for short
                    capital += position_value  # Get cash from short sale
                    position_entry_price = current_price
                    trades.append({
                        'timestamp': df.iloc[i]['Timestamp'],
                        'type': 'SHORT',
                        'price': current_price,
                        'shares': shares,
                        'value': position_value,
                        'reason': 'Lorentzian Signal'
                    })
        
        # Calculate portfolio value
        portfolio_value = capital + (position * current_price)
        portfolio_values.append(portfolio_value)
    
    # Final liquidation
    if position != 0:
        final_value = abs(position) * df.iloc[-1]['Close']
        if position > 0:
            capital += final_value
        else:
            capital -= final_value
        
        trades.append({
            'timestamp': df.iloc[-1]['Timestamp'],
            'type': 'LIQUIDATE',
            'price': df.iloc[-1]['Close'],
            'shares': abs(position),
            'value': final_value,
            'reason': 'End of Period'
        })
    
    final_portfolio_value = capital
    
    return {
        'final_capital': capital,
        'final_portfolio_value': final_portfolio_value,
        'total_return': (final_portfolio_value - initial_capital) / initial_capital,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'max_portfolio_value': max_portfolio_value
    }

def calculate_advanced_metrics(results, df):
    """Calculate comprehensive performance metrics"""
    print("📊 Calculating advanced performance metrics...")
    
    portfolio_values = np.array(results['portfolio_values'])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]
    
    # Basic metrics
    total_return = results['total_return']
    periods_per_year = 252 * 48 / 30  # 30-min bars per year
    annualized_return = (1 + total_return) ** (periods_per_year / len(df)) - 1
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Downside deviation for Sortino ratio
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Drawdown analysis
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Trade analysis
    trades = results['trades']
    num_trades = len([t for t in trades if t['type'] in ['BUY', 'SHORT']])
    
    # Calculate trade P&L
    trade_pnl = []
    position_trades = []
    
    for trade in trades:
        if trade['type'] in ['BUY', 'SHORT']:
            position_trades.append(trade)
        elif trade['type'] in ['SELL (Risk Mgmt)', 'COVER (Risk Mgmt)', 'LIQUIDATE']:
            if position_trades:
                entry_trade = position_trades[-1]
                if entry_trade['type'] == 'BUY':
                    pnl = (trade['price'] - entry_trade['price']) * entry_trade['shares']
                else:  # SHORT
                    pnl = (entry_trade['price'] - trade['price']) * entry_trade['shares']
                
                pnl_pct = pnl / (entry_trade['price'] * entry_trade['shares'])
                trade_pnl.append(pnl_pct)
                position_trades = []
    
    # Trade statistics
    if trade_pnl:
        win_rate = sum(1 for pnl in trade_pnl if pnl > 0) / len(trade_pnl)
        avg_win = np.mean([pnl for pnl in trade_pnl if pnl > 0]) if any(pnl > 0 for pnl in trade_pnl) else 0
        avg_loss = np.mean([pnl for pnl in trade_pnl if pnl < 0]) if any(pnl < 0 for pnl in trade_pnl) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        avg_trade = np.mean(trade_pnl)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_trade = 0
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Sortino Ratio': f"{sortino_ratio:.2f}",
        'Calmar Ratio': f"{calmar_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Number of Trades': num_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Average Win': f"{avg_win:.2%}",
        'Average Loss': f"{avg_loss:.2%}",
        'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
        'Average Trade': f"{avg_trade:.2%}"
    }
    
    return metrics

def main():
    """Main execution function"""
    print("🎯 OPTIMIZED LORENTZIAN STRATEGY BACKTEST")
    print("=" * 60)
    
    # File path
    data_path = Path("/home/QuantNova/GrandModel/colab/data/NQ - 30 min - ETH.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Load and prepare data
    df = load_and_clean_data(data_path)
    if df is None:
        return
    
    # Calculate advanced indicators
    df = calculate_advanced_indicators(df)
    
    # Calculate Lorentzian features
    df, feature_columns = calculate_lorentzian_features(df)
    
    # Generate optimized signals
    df = generate_lorentzian_signals(df, feature_columns, lookback=150, k_neighbors=8)
    
    # Run optimized backtest
    results = run_optimized_backtest(df)
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(results, df)
    
    # Display results
    print("\n🏆 OPTIMIZED BACKTEST RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value}")
    
    print(f"\nInitial Capital    : ${100000:,.2f}")
    print(f"Final Portfolio    : ${results['final_portfolio_value']:,.2f}")
    print(f"Total Profit/Loss  : ${results['final_portfolio_value'] - 100000:,.2f}")
    print(f"Peak Portfolio     : ${results['max_portfolio_value']:,.2f}")
    
    # Performance assessment against targets
    print("\n🎯 PERFORMANCE TARGET ASSESSMENT")
    print("=" * 60)
    
    sharpe = float(metrics['Sharpe Ratio'])
    max_dd = float(metrics['Maximum Drawdown'].rstrip('%')) / 100
    win_rate = float(metrics['Win Rate'].rstrip('%')) / 100
    calmar = float(metrics['Calmar Ratio'])
    sortino = float(metrics['Sortino Ratio'])
    
    targets = [
        ("Sharpe Ratio > 2.0", sharpe > 2.0, f"{sharpe:.2f}"),
        ("Max Drawdown < 15%", abs(max_dd) < 0.15, f"{abs(max_dd):.1%}"),
        ("Win Rate > 60%", win_rate > 0.60, f"{win_rate:.1%}"),
        ("Calmar Ratio > 1.0", calmar > 1.0, f"{calmar:.2f}"),
        ("Sortino Ratio > 2.5", sortino > 2.5, f"{sortino:.2f}"),
        ("Positive Returns", results['total_return'] > 0, f"{results['total_return']:.1%}")
    ]
    
    passed_tests = 0
    total_tests = len(targets)
    
    for target_name, passed, value in targets:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{target_name:25}: {status} ({value})")
        if passed:
            passed_tests += 1
    
    print(f"\nTarget Achievement: {passed_tests}/{total_tests} targets met")
    
    # Overall assessment
    if passed_tests >= 5:
        print("🎉 STRATEGY PERFORMANCE: EXCELLENT")
    elif passed_tests >= 4:
        print("🚀 STRATEGY PERFORMANCE: GOOD")
    elif passed_tests >= 3:
        print("⚠️ STRATEGY PERFORMANCE: FAIR")
    else:
        print("🔴 STRATEGY PERFORMANCE: NEEDS IMPROVEMENT")
    
    # Show some sample trades
    print(f"\n📋 SAMPLE TRADES (Last 10)")
    print("=" * 60)
    recent_trades = results['trades'][-10:]
    for trade in recent_trades:
        print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
              f"{trade['type']:15} | ${trade['price']:8.2f} | "
              f"{trade['shares']:8.1f} | ${trade['value']:10,.0f}")
    
    print("\n✅ Optimized backtest completed successfully!")

if __name__ == "__main__":
    main()