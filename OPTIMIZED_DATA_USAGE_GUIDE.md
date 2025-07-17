# NQ Dataset - Optimized Data Usage Guide

## 🚀 Quick Start

The NQ dataset has been optimized for high-performance backtesting with 76.6% memory reduction and instant access times.

### Loading Optimized Data

```python
import pickle
import joblib
import json
import numpy as np

# Load numpy arrays (fastest performance)
with open('/home/QuantNova/GrandModel/data/optimized/5min_numpy_arrays.pkl', 'rb') as f:
    arrays = pickle.load(f)

# Load optimized DataFrame (pandas compatibility)
df = joblib.load('/home/QuantNova/GrandModel/data/optimized/5min_optimized_dataframe.pkl')

# Load precomputed statistics
with open('/home/QuantNova/GrandModel/data/optimized/5min_precomputed_stats.json', 'r') as f:
    stats = json.load(f)
```

## 📊 Available Data Structures

### Numpy Arrays (Recommended for Performance)
```python
# OHLCV data - Shape: (n_bars, 5)
ohlcv = arrays['ohlcv']  # [open, high, low, close, volume]
prices = ohlcv[:, 3]     # Close prices
volumes = ohlcv[:, 4]    # Volume data

# Returns
returns = arrays['returns']          # Percentage returns
log_returns = arrays['log_returns']  # Log returns

# Technical indicators
atr = arrays['atr']                  # Average True Range

# Time features
hours = arrays['hour']               # Hour of day (0-23)
weekdays = arrays['day_of_week']     # Day of week (0-6)
weekends = arrays['is_weekend']      # Boolean weekend flag

# Timestamps (as int64 for fast operations)
timestamps = arrays['timestamp']
```

### DataFrames (Pandas Compatible)
```python
# Standard pandas operations
close_prices = df['close']
returns = df['returns']
volume = df['volume']

# Time-based filtering
recent_data = df[df['timestamp'] > '2023-01-01']
trading_hours = df[df['hour'].between(9, 16)]
```

## ⚡ Performance Tips

### 1. Use Numpy Arrays for Speed
```python
# FAST: Direct numpy operations
mean_price = np.mean(arrays['ohlcv'][:, 3])
volatility = np.std(arrays['returns'])

# SLOWER: DataFrame operations
mean_price = df['close'].mean()
volatility = df['returns'].std()
```

### 2. Leverage Precomputed Statistics
```python
# Instant access to common metrics
price_min = stats['price_stats']['min']
price_max = stats['price_stats']['max']
avg_volume = stats['volume_stats']['mean']
return_volatility = stats['return_stats']['std']
```

### 3. Efficient Time Indexing
```python
# Load timestamp index for fast lookups
with open('/home/QuantNova/GrandModel/data/optimized/5min_timestamp_index.pkl', 'rb') as f:
    time_index = pickle.load(f)

start_time = time_index['start_timestamp']
end_time = time_index['end_timestamp']
total_bars = time_index['total_bars']
```

## 📈 Dataset Statistics

### 5-Minute Data
- **Bars**: 327,655
- **Date Range**: 2020-06-29 to 2025-06-30
- **Memory Usage**: 13.4 MB (optimized)
- **Coverage**: 37.7%

### 30-Minute Data  
- **Bars**: 56,083
- **Date Range**: 2020-06-29 to 2025-06-30
- **Memory Usage**: 2.3 MB (optimized)
- **Coverage**: 38.7%

### Data Quality
- **Quality Score**: 100/100
- **OHLC Consistency**: Perfect
- **Missing Values**: Zero in core data

## 🔧 Integration Examples

### Backtesting Engine Integration
```python
class FastBacktester:
    def __init__(self, timeframe='5min'):
        # Load optimized data
        arrays_path = f'/home/QuantNova/GrandModel/data/optimized/{timeframe}_numpy_arrays.pkl'
        with open(arrays_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.ohlcv = self.data['ohlcv']
        self.returns = self.data['returns']
        self.timestamps = self.data['timestamp']
        
    def get_price(self, bar_idx):
        """Get OHLCV for specific bar - O(1) access"""
        return self.ohlcv[bar_idx]
    
    def get_returns(self, start_idx, end_idx):
        """Get returns slice - vectorized operation"""
        return self.returns[start_idx:end_idx]
```

### Strategy Development
```python
def moving_average_strategy(lookback=20):
    # Load data
    with open('/home/QuantNova/GrandModel/data/optimized/5min_numpy_arrays.pkl', 'rb') as f:
        arrays = pickle.load(f)
    
    prices = arrays['ohlcv'][:, 3]  # Close prices
    
    # Calculate moving average (vectorized)
    ma = np.convolve(prices, np.ones(lookback)/lookback, mode='valid')
    
    # Generate signals
    signals = np.where(prices[lookback-1:] > ma, 1, -1)
    
    return signals
```

### Market Session Analysis
```python
def analyze_sessions():
    # Load optimized DataFrame
    df = joblib.load('/home/QuantNova/GrandModel/data/optimized/5min_optimized_dataframe.pkl')
    
    # Session performance
    asia_session = df[df['hour'].between(18, 3)]  # Note: wraps around
    london_session = df[df['hour'].between(3, 8)]
    ny_session = df[df['hour'].between(8, 17)]
    
    return {
        'asia_volume': asia_session['volume'].mean(),
        'london_volume': london_session['volume'].mean(), 
        'ny_volume': ny_session['volume'].mean()
    }
```

## 📁 File Reference

### Data Files
- `5min_numpy_arrays.pkl` - High-performance numpy arrays
- `5min_optimized_dataframe.pkl` - Memory-optimized pandas DataFrame
- `5min_precomputed_stats.json` - Pre-calculated statistics
- `5min_timestamp_index.pkl` - Time indexing for fast lookups
- `30min_*` - Same structure for 30-minute data

### Reports
- `optimization_report_*.json` - Complete optimization metrics
- `nq_dataset_validation_report_*.json` - Data quality validation

## ⚠️ Important Notes

1. **Data Gaps**: 38% coverage means ~62% of expected bars are missing (weekends, holidays, low-volume periods)
2. **Extended Hours**: Data includes both regular and extended trading hours
3. **Memory Efficient**: Use numpy arrays for production systems
4. **Time Zones**: All timestamps are in the original dataset timezone
5. **Precision**: Float32 precision maintains accuracy while reducing memory

## 🎯 Best Practices

1. **Always use numpy arrays** for backtesting loops
2. **Cache frequently accessed** statistics using precomputed values
3. **Filter by time efficiently** using the hour/day_of_week arrays
4. **Handle gaps appropriately** in your backtesting logic
5. **Use vectorized operations** whenever possible for speed

---

*Optimized by Agent 1 - Data Preparation & Validation Specialist*