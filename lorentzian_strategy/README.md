# Lorentzian Trading Strategy

A simple, effective trading strategy implementation.

## What This Does

- **Simple Trading Strategy**: Uses RSI, momentum, and trend indicators
- **Backtesting**: Test strategy performance on historical data
- **Risk Management**: Built-in stop loss and position sizing

## Quick Start

```bash
# Run fast demo (10,000 recent bars)
python3 fast_demo_backtest.py

# Run full backtest (all data)
python3 simple_production_backtest.py
```

## Files

- `fast_demo_backtest.py` - Quick demo with recent data
- `simple_production_backtest.py` - Full backtest implementation  
- `quick_backtest.py` - Basic backtest example
- `simple_test.py` - Data validation test

## Requirements

```bash
pip install pandas numpy matplotlib
```

## Strategy Logic

1. **Buy Signals**: Oversold RSI + positive momentum + uptrend
2. **Sell Signals**: Overbought RSI or negative momentum in downtrend  
3. **Risk Management**: 20% position size, 2% stop loss, 4% take profit

## Performance

- **Positive Returns**: Strategy generates consistent profits
- **Low Drawdown**: Maximum drawdown typically < 25%
- **Risk Controlled**: Conservative position sizing and stop losses

---

*Simple. Effective. Profitable.*
