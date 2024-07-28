import numpy as np
import pandas as pd
from ib_insync import *
from Interface import IBInterface  
from Config import get_config

config = get_config()

def connect_ibkr():
    """
    Connect to Interactive Brokers using the custom interface.
    
    Returns:
    IBInterface: Connected IB interface instance
    """
    ib_interface = IBInterface(config.paper_api_port if config.use_paper_trading else config.live_api_port, config.account_id)
    if ib_interface.connect():
        print("Successfully connected to IBKR")
        return ib_interface
    else:
        print("Failed to connect to IBKR. Make sure TWS or Gateway is running and Allow API connections is enabled")
        return None

def get_historical_data(ib_interface, symbol, duration='5 D', bar_size='1 hour'):
    """
    Fetch historical data from IBKR using the custom interface.
    
    Args:
    ib_interface (IBInterface): Connected IB interface instance
    symbol (str): Forex pair symbol (e.g., 'EURUSD')
    duration (str): Duration of historical data
    bar_size (str): Bar size (e.g., '1 hour', '5 mins')
    
    Returns:
    pd.DataFrame: Historical data
    """
    contract = ib_interface.create_contract(symbol)
    df = ib_interface.get_historical_data(contract, duration, bar_size)
    return df

def calculate_features(df):
    """
    Calculate features for Lorentzian Classification.
    
    Args:
    df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
    pd.DataFrame: DataFrame with calculated features
    """
    df['RSI'] = calculate_rsi(df['close'])
    df['ADX'] = calculate_adx(df['high'], df['low'], df['close'])
    df['CCI'] = calculate_cci(df['high'], df['low'], df['close'])
    df['WT1'], df['WT2'] = calculate_wt((df['high'] + df['low'] + df['close']) / 3)
    
    if config.lorentzian.use_remote_fractals:
        df['Bullish_Fractal'], df['Bearish_Fractal'] = calculate_fractal(df['high'], df['low'])
    
    return df.dropna()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/window).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (window - 1)) + dx) / window
    adx_smooth = adx.ewm(alpha=1/window).mean()
    return adx_smooth

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: pd.Series(x).mad())
    cci = (tp - sma) / (0.015 * mad)
    return cci

def calculate_wt(hlc3, channel_length=10, average_length=21):
    esa = hlc3.ewm(span=channel_length).mean()
    d = (hlc3 - esa).abs().ewm(span=channel_length).mean()
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ci.ewm(span=average_length).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2

def calculate_fractal(high, low, window=5):
    def is_bullish_fractal(x):
        return x[window//2] == max(x)
    
    def is_bearish_fractal(x):
        return x[window//2] == min(x)
    
    bullish_fractals = high.rolling(window=window, center=True).apply(is_bullish_fractal)
    bearish_fractals = low.rolling(window=window, center=True).apply(is_bearish_fractal)
    
    return bullish_fractals, bearish_fractals

def calculate_lorentzian_distance(x1, x2):
    return np.log(1 + np.sum(np.abs(x1 - x2)))

def calculate_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def calculate_distance(x1, x2):
    if config.lorentzian.reduce_warping:
        return calculate_euclidean_distance(x1, x2)
    else:
        return calculate_lorentzian_distance(x1, x2)

def calculate_position_size(account_balance):
    return int(account_balance * config.risk_management.max_risk_per_trade / config.risk_management.stop_loss_percent)

def apply_downsampling(df):
    if config.lorentzian.use_downsampling:
        return df.iloc[::4]  # Select every 4th row
    return df

def get_feature_columns():
    columns = ['RSI', 'ADX', 'CCI', 'WT1', 'WT2']
    if config.lorentzian.use_remote_fractals:
        columns.extend(['Bullish_Fractal', 'Bearish_Fractal'])
    return columns[:config.lorentzian.feature_count]

def prepare_features(df):
    df_features = calculate_features(df)
    df_features = apply_downsampling(df_features)
    feature_columns = get_feature_columns()
    X = df_features[feature_columns]
    y = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    return X, y[:-1]  # Remove the last row of y since we don't have a target for the last data point

def get_symbols():
    return config.trading.symbols

def get_timeframes():
    return config.trading.timeframes

def get_trading_hours():
    return config.trading.trading_start_time, config.trading.trading_end_time